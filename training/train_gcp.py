import argparse
import json
import os
import sys
import torch
import torchaudio
import tqdm

from meld_dataset import prepare_dataloaders
from models import MultiModalSentimentModel, MultiModalTrainer
from install_ffmpeg import install_ffmpeg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.001)

    # These will be passed by GCP job config
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)

    return parser.parse_args()


def main():
    # Install FFMPEG
    if not install_ffmpeg():
        print("FFmpeg installation failed. Please check your internet connection and try again.")
        sys.exit(1)

    print("Available Audio Backends:")
    print(str(torchaudio.list_audio_backends()))

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = prepare_dataloaders(
        train_csv=os.path.join(args.train_dir, 'train_sent_emo.csv'),
        train_video_dir=os.path.join(args.train_dir, 'train_splits'),
        dev_csv=os.path.join(args.val_dir, 'dev_sent_emo.csv'),
        dev_video_dir=os.path.join(args.val_dir, 'dev_splits_complete'),
        test_csv=os.path.join(args.test_dir, 'test_sent_emo.csv'),
        test_video_dir=os.path.join(args.test_dir, 'output_repeated_splits_test'),
        batch_size=args.batch_size
    )

    model = MultiModalSentimentModel().to(device)
    trainer = MultiModalTrainer(model, train_loader, val_loader)

    best_val_loss = float('inf')

    metrics_data = {
        'train_losses': [],
        'val_losses': [],
        "epochs": [],
    }

    for epoch in tqdm.tqdm(range(args.epochs), desc="Epochs"):
        train_loss = trainer.train_epoch()
        val_loss, val_metrics = trainer.evaluate(val_loader)

        metrics_data['train_losses'].append(train_loss["total"])
        metrics_data['val_losses'].append(val_loss["total"])
        metrics_data['epochs'].append(epoch)

        # Print metrics for logging
        print(json.dumps({
            "train_loss": train_loss["total"],
            "val_loss": val_loss["total"],
            "emotion_precision": val_metrics["emotion_precision"],
            "emotion_accuracy": val_metrics["emotion_accuracy"],
            "sentiment_precision": val_metrics["sentiment_precision"],
            "sentiment_accuracy": val_metrics["sentiment_accuracy"],
        }))

        if val_loss["total"] < best_val_loss:
            best_val_loss = val_loss["total"]
            torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))

    print("Evaluating on test set...")
    test_loss, test_metrics = trainer.evaluate(test_loader, phase='test')

    print(json.dumps({
        "test_loss": test_loss["total"],
        "test_emotion_accuracy": test_metrics["emotion_accuracy"],
        "test_sentiment_accuracy": test_metrics["sentiment_accuracy"],
        "test_emotion_precision": test_metrics["emotion_precision"],
        "test_sentiment_precision": test_metrics["sentiment_precision"],
    }))


if __name__ == "__main__":
    main()
