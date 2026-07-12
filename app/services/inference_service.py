from app.schemas.inference import InferenceRequest, InferenceResponse


class InferenceService:
    def predict(self, payload: InferenceRequest) -> InferenceResponse:
        score = sum(payload.features.values()) / max(len(payload.features), 1)
        label = "normal" if score < 0.7 else "attention"

        return InferenceResponse(
            route_id=payload.route_id,
            score=round(score, 4),
            label=label,
        )
