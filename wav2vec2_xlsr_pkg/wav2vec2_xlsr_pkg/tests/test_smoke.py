def test_imports():
    import wav2vec2_xlsr
    from wav2vec2_xlsr.models.asr import build_asr
    b = build_asr("facebook/wav2vec2-base-960h")
    assert b.model is not None and b.processor is not None
