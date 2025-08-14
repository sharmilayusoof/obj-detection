"""Generic SM inference.py for all TensorFlow inference tasks.

Read metadata and dispatch to task-specifc pipelines
https://sagemaker.readthedocs.io/en/stable/frameworks/tensorflow/using_tf.html#deploying-directly
-from-model-artifacts
https://github.com/aws/sagemaker-tensorflow-serving-container
https://sagemaker.readthedocs.io/en/stable/frameworks/tensorflow/deploying_tensorflow_serving
.html
https://cloud.google.com/blog/topics/developers-practitioners/add-preprocessing-functions-tensorflow-models-and-deploy-vertex-ai
https://towardsdatascience.com/serving-image-based-deep-learning-models-with-tensorflow-servings-restful-api-d365c16a7dc4
"""


from container_setup import initialize_tf_requirements


initialize_tf_requirements()

import json
import re
from typing import List
from typing import Tuple

import numpy as np
import requests
from constants import constants
from PIL import Image
from six import BytesIO


LABELS_INFO = "/opt/ml/model/labels_info.json"
PROBABILITIES = "probabilities"
LABELS = "labels"
PREDICTED_LABEL = "predicted_label"
PREDICTIONS = "predictions"
VERBOSE_EXTENSION = ";verbose"
REQUEST_CONTENT_TYPE = "application/x-image"
STR_DECODE_CODE = "utf-8"


with open(LABELS_INFO, "r") as txtFile:
    labels = json.loads(txtFile.read())[LABELS]


def handler(data, context):
    """Handle request.

    Args:
        data (obj): the request data
        context (Context): an object containing request and configuration details
    Returns:
        (bytes, string): data to return to client, (optional) response content type
    """
    processed_input = _process_input(data, context)
    response = requests.post(context.rest_uri, data=processed_input)
    return _process_output(response, context)


def _process_input(data, context) -> str:
    """Encode input data to base64.

    Args:
        data (obj): the request data
        context (Context): an object containing request and configuration details
    Returns:
        string: data for input to model.
    Raises:
        ValueError: if context.request_content_type is not the expected REQUEST_CONTENT_TYPE
    """

    if context.request_content_type == REQUEST_CONTENT_TYPE:
        # pass through json (assumes it's correctly formed)
        img = Image.open(BytesIO(data.read())).convert("RGB")
        (im_width, im_height) = img.size
        img_np = np.array(img.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
        img_input = np.expand_dims(img_np, axis=0)
        img = {"instances": img_input}
        img = json.dumps(
            {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in img.items()}
        )
        return str(img)
    raise ValueError('{{"error": "unsupported content type {}"}}'.format(context.request_content_type or "unknown"))


def get_top_k_predictions(
    predictions: [np.array, np.array, np.array], k: int
) -> Tuple[List[np.array], List[int], List[float]]:
    """Return the top k predictions with highest scores."""

    boxes, labels, scores = predictions
    k = min(k, len(boxes))
    top_k_indices = np.argpartition(scores, -k)[-k:]
    boxes_top_k = [boxes[int(idx)] for idx in top_k_indices]
    labels_top_k = [labels[int(idx)] for idx in top_k_indices]
    scores_top_k = [scores[int(idx)] for idx in top_k_indices]
    return boxes_top_k, labels_top_k, scores_top_k


def _process_output(data, context):
    """Take the response of processed image input, return dictionary with predictions.

    Args:
        data (obj): the response of processed image input
        context (Context): an object containing request and configuration details
    Returns:
        (string, string): data to return to client, (optional) response content type
    """
    if data.status_code != 200:
        raise ValueError(data.content.decode(constants.STR_DECODE_CODE))
    response_content_type = context.accept_header
    model_output = json.loads(data.content)
    model_prediction = model_output[constants.PREDICTIONS][0]

    # Transposing the detection boxes to keep it consistent with other framework outputs.
    detection_boxes_transposed = [
        [detected_box[1], detected_box[0], detected_box[3], detected_box[2]]
        for detected_box in model_prediction[constants.DETECTION_BOXES]
    ]
    normalized_predictions = (
        detection_boxes_transposed,
        model_prediction[constants.DETECTION_CLASSES],
        model_prediction[constants.DETECTION_SCORES],
    )

    # If only a subset of predictions are requested, we return the top k predictions as per scores
    n_predictions_pattern_search = re.search(f"{constants.N_PREDICTIONS_KEY}=[0-9]*", response_content_type)
    if n_predictions_pattern_search is not None:
        n_prediction_pattern = n_predictions_pattern_search.group(0)
        n_output_predictions = int(n_prediction_pattern.lstrip(f"{constants.N_PREDICTIONS_KEY}="))
        response_content_type = response_content_type.replace(n_prediction_pattern, "")
        normalized_predictions = get_top_k_predictions(normalized_predictions, k=n_output_predictions)
    else:
        normalized_predictions = get_top_k_predictions(normalized_predictions, k=constants.DEFAULT_N_PREDICTIONS)
    output = {
        constants.NORMALIZED_BOXES: normalized_predictions[0],
        constants.CLASSES: normalized_predictions[1],
        constants.SCORES: normalized_predictions[2],
    }

    # If the response_content_type is verbose, we return the labels as well as the original model output.
    if constants.VERBOSE_EXTENSION in response_content_type:
        output[constants.LABELS] = labels
        response_content_type = response_content_type.replace(constants.VERBOSE_EXTENSION, "")
        output[constants.TF_MODEL_OUTPUT] = model_output

    return json.dumps(output), response_content_type
