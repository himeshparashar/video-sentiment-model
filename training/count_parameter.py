from models import MultimodalSentimentModel

def count_parameters(model):
    params_dict = {
        'text_encoder': {'trainable': 0, 'non_trainable': 0},
        'video_encoder': {'trainable': 0, 'non_trainable': 0},
        'audio_encoder': {'trainable': 0, 'non_trainable': 0},
        'fusion_layer': {'trainable': 0, 'non_trainable': 0},
        'emotion_classifier': {'trainable': 0, 'non_trainable': 0},
        'sentiment_classifier': {'trainable': 0, 'non_trainable': 0}
    }

    total_trainable = 0
    total_non_trainable = 0

    for name, param in model.named_parameters():
        param_count = param.numel()
        
        if param.requires_grad:
            total_trainable += param_count
            param_type = 'trainable'
        else:
            total_non_trainable += param_count
            param_type = 'non_trainable'

        if 'text_encoder' in name:
            params_dict['text_encoder'][param_type] += param_count
        elif 'video_encoder' in name:
            params_dict['video_encoder'][param_type] += param_count
        elif 'audio_encoder' in name:
            params_dict['audio_encoder'][param_type] += param_count
        elif 'fusion_layer' in name:
            params_dict['fusion_layer'][param_type] += param_count
        elif 'emotion_classifier' in name:
            params_dict['emotion_classifier'][param_type] += param_count
        elif 'sentiment_classifier' in name:
            params_dict['sentiment_classifier'][param_type] += param_count
        
    return params_dict, total_trainable, total_non_trainable

if __name__ == "__main__":
    model = MultimodalSentimentModel()
    params_dict, total_trainable, total_non_trainable = count_parameters(model)
    
    print("Parameter count by component:")
    for component, counts in params_dict.items():
        print(f"\n{component}:")
        print(f"  Trainable parameters:     {counts['trainable']:,}")
        print(f"  Non-trainable parameters: {counts['non_trainable']:,}")
    
    print(f"\nTotal trainable parameters:     {total_trainable:,}")
    print(f"Total non-trainable parameters: {total_non_trainable:,}")
    print(f"Total parameters:               {total_trainable + total_non_trainable:,}")
