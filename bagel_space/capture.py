"""Capture the first block output of a selected BAGEL inference stage."""


class Capture:
    def __init__(self, layers, indices, stage="text"):
        self.layers = layers
        self.indices = list(dict.fromkeys(indices))
        self.stage = stage
        self.states = {}
        self.alignment = {}
        self.handles = []

    def __enter__(self):
        self.states.clear()
        self.alignment.clear()
        for index in self.indices:
            self.handles.append(self.layers[index].register_forward_hook(
                self._hook(index), with_kwargs=True,
            ))
        return self

    def _hook(self, index):
        def save(module, args, kwargs, output):
            stage = "generation" if kwargs["mode"] == "gen" else (
                "text" if kwargs["is_causal"] else "image"
            )
            if stage != self.stage or index in self.states:
                return
            self.states[index] = output[0].detach().cpu().clone()
            if not self.alignment:
                self.alignment = dict(
                    stage=stage, query_indexes=kwargs["packed_query_indexes"].cpu().tolist(),
                    query_lengths=kwargs["query_lens"].cpu().tolist(),
                    cached_lengths=kwargs["key_values_lens"].cpu().tolist(),
                )
                if stage == "generation":
                    self.alignment.update(
                        visual_indexes=kwargs["packed_vae_token_indexes"].cpu().tolist(),
                        special_indexes=kwargs["packed_text_indexes"].cpu().tolist(),
                    )
                if stage == "image":
                    self.alignment["visual_span"] = [1, output[0].shape[0] - 1]
        return save

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
