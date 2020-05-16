from accuracy_checker.adapters import Adapter
from accuracy_checker.config import ConfigValidator, StringField
from accuracy_checker.representation import MultiLabelRecognitionAnnotation, MultiLabelRecognitionPrediction
import openvino.tools.calibration as calibration

class ChestXAdapter(Adapter):
    __provider__ = 'chest_xray'
    prediction_types = (MultiLabelRecognitionPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'attributes_recognition_out': StringField(
                description="Output layer name for attributes recognition.", optional=True)
        })
        return parameters

    def validate_config(self):
        super().validate_config(on_extra_argument=ConfigValidator.IGNORE_ON_EXTRA_ARGUMENT)

    def configure(self):
        self.attributes_recognition_out = self.launcher_config.get('attributes_recognition_out', self.output_blob)

    def process(self, raw, identifiers=None, frame_meta=None):
        result = []
        raw_output = self._extract_predictions(raw, frame_meta)
        self.attributes_recognition_out = self.attributes_recognition_out or self.output_blob
        for identifier, multi_label in zip(identifiers, raw_output[self.attributes_recognition_out]):
            multi_label[multi_label > 0.5] = 1.
            multi_label[multi_label <= 0.5] = 0.
            result.append(MultiLabelRecognitionPrediction(identifier, multi_label.reshape(-1)))
        return result

if __name__ == '__main__':
    with calibration.CommandLineProcessor.process() as config:
        network = calibration.Calibrator(config).run()
        if network:
            network.serialize(config.output_model)
