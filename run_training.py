from src.train import train_model

if __name__ == "__main__":
    model, class_names = train_model(
        data_dir="data/train",
        num_classes=20,
        model_name="efficientnet_b0",
        epochs=10,
        batch_size=32,
        lr=1e-4
    )

    print("✅ Training complete. Model saved to models/efficientnet_b0.pth")
