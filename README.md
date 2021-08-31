# AdaSpeech 2: Adaptive Text to Speech with Untranscribed Data [WIP]
Unofficial Pytorch implementation of [AdaSpeech 2](https://arxiv.org/abs/2104.09715).


![](./assets/model.PNG)


## Requirements :
All code written in `Python 3.6.2` .
* Install Pytorch
> Before installing pytorch please check your Cuda version by running following command : 
`nvcc --version`
```commandline
pip install torch torchvision
```
In this repo I have used Pytorch 1.6.0 for `torch.bucketize` feature which is not present in previous versions of PyTorch.


* Installing other requirements :
```commandline
pip install -r requirements.txt
```

* To use Tensorboard install `tensorboard version 1.14.0` seperatly with supported `tensorflow (1.14.0)`



## For Preprocessing :

`filelists` folder contains MFA (Motreal Force aligner) processed LJSpeech dataset files so you don't need to align text with audio (for extract duration) for LJSpeech dataset.
For other dataset follow instruction [here](https://github.com/ivanvovk/DurIAN#6-how-to-align-your-own-data). For other pre-processing run following command :
```commandline
python nvidia_preprocessing.py -d path_of_wavs
```
For finding the min and max of F0 and Energy
```commandline
python compute_statistics.py
```
Update the following in `hparams.py` by min and max of F0 and Energy
```
p_min = Min F0/pitch
p_max = Max F0
e_min = Min energy
e_max = Max energy
```

## Training :

[WIP]

## Note :
* For more complete and end to end Voice cloning or Text to Speech (TTS) toolbox please visit [Deepsync Technologies](https://deepsync.co/).

## Citations :
```bibtex
@misc{chen2021adaspeech,
      title={AdaSpeech: Adaptive Text to Speech for Custom Voice}, 
      author={Mingjian Chen and Xu Tan and Bohan Li and Yanqing Liu and Tao Qin and Sheng Zhao and Tie-Yan Liu},
      year={2021},
      eprint={2103.00993},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```
```bibtex
@misc{yan2021adaspeech,
      title={AdaSpeech 2: Adaptive Text to Speech with Untranscribed Data}, 
      author={Yuzi Yan and Xu Tan and Bohan Li and Tao Qin and Sheng Zhao and Yuan Shen and Tie-Yan Liu},
      year={2021},
      eprint={2104.09715},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```


