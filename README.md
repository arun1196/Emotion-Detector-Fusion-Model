# Emotion Detector Fusion Model
Part of the project for A Multi-Modal Deep Learning Framework for Real-Time Biometric Spoof Detection Using Face and Voice Fusion

# Generate the Fusion Model

We use the dataset from The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)

The data is publicy available at https://zenodo.org/records/1188976

Download the RAVDESS dataset from above and set it at the data DIR `./data/RAVDESS/...` 

Run the following to create your fusion model that can detect emotion based on audio/video

```bash
python scripts/train.py \
  --audio-dir data/RAVDESS/Audio_Speech_Actors_01-04 \
  --video-dir data/RAVDESS/Video_Speech_Actors_01-04 \                                
  --fusion hybrid \                                                                   
  --epochs 10 \                            
  --batch-size 8 \ 
  --lr 1e-3 \                        
  --output-dir outputs/hybrid
```

The output dir is outputs/hybrid and this is where your new `fusion_model.pth` will be saved.

### Notes
You can use as many speech/video actor content to train but remenber the more data you use, the larger and time consuming process it will be. 

# Using the Fusion Model

Once you generate your fusion model with above instructions, you can pass new content to the model to use it. This is fairly fast once the model has been created. We specify the model we want to use in our command with the data.

Currently our list of emotions are: `"calm","happy","sad","angry","fearful","surprised" and "disgust"`

A command to evaluate emption from the provide list of emotions can look like:

```bash
export PYTHONPATH="$PWD" 
python infer.py \                                     
  --audio "data/RAVDESS/Audio_Speech_Actors_01-04/Actor_01/03-01-07-01-01-02-01.wav" \
  --video "data/RAVDESS/Video_Speech_Actors_01-04/Actor_01/01-01-02-01-02-01-01.mp4" \
  --model outputs/hybrid/fusion_model.pth \
  --fusion hybrid \
  --config configs/fusion_config.yaml
```

This will output the emotion like `Predicted emotion/spoof: calm`