"""Replaceable mock multimodal operators."""
class MockOperator:
    name = "operator"
    result = {}

    def run(self, payload):
        print(f"[MOCK operator:{self.name}] invoked")
        return self.result.copy() if isinstance(self.result, dict) else list(self.result)


class ShotSegmentOperator(MockOperator):
    name, result = "shot_segment", [{"start": 0.0, "end": 3.0, "transition_type": "cut"}]

class VLMTagOperator(MockOperator):
    name, result = "vlm_tag", {"shot_size": "close-up", "subjects": [], "actions": []}

class ASROperator(MockOperator):
    name, result = "asr", {"text": "", "segments": [], "pace": "normal"}

class OCROperator(MockOperator):
    name, result = "ocr", {"frames": []}

class EmotionOperator(MockOperator):
    name, result = "emotion", {"speaker_id": "unknown", "emotion_curve": []}

class SceneEventOperator(MockOperator):
    name, result = "scene_event", {"scenes": []}


REGISTRY = {
    "shot_segment": ShotSegmentOperator(), "vlm_tag": VLMTagOperator(),
    "asr": ASROperator(), "ocr": OCROperator(), "emotion": EmotionOperator(),
    "scene_event": SceneEventOperator(),
}


def run(name, payload):
    if name not in REGISTRY:
        raise KeyError(f"unknown operator: {name}")
    return REGISTRY[name].run(payload)
