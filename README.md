# Neural Digit Classification

The TensorFlow digit-classification Colab rebuilt around explicit preprocessing, a compact CNN factory, deterministic seeds, and testable input validation.

```bash
pip install -e .[train,dev]
pytest
```

The library accepts 28×28 grayscale arrays and converts integer labels only at the training boundary. Save held-out metrics and confusion matrices separately from training history.

