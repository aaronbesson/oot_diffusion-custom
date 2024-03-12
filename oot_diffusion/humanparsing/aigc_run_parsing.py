from .parsing_api import load_atr_model, load_lip_model, inference


class Parsing:
    def __init__(self, device, hg_root: str):
        self.device = device
        self.atr_model = load_atr_model(hg_root)
        self.lip_model = load_lip_model(hg_root)

    def __call__(self, input_image):
        parsed_image, face_mask = inference(self.atr_model, self.lip_model, input_image)
        return parsed_image, face_mask
