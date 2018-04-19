# Convolutional Neural Networks to Enhance Coded Speech
(Here Part of the project code，**Not for commercial use!!!**) 
  
**Abstract**—Enhancing coded speech suffering from far-end acoustic background noise, quantization noise, and potentially transmission errors, is a challenging task. In this work we propose two postprocessing approaches applying convolutional neural networks (CNNs) either in the time domain or the cepstral domain to enhance the coded speech without any modification of the codecs. The time domain approach follows an end-to-end fashion, while the cepstral domain approach uses analysis-synthesis with
cepstral domain features. The proposed postprocessors in both domains are evaluated for various narrowband and wideband speech codecs in a wide range of conditions. The proposed postprocessor improves speech quality (PESQ) by up to 0.25 MOS-LQO points for G.711, 0.30 points for G.726, 0.82 points for G.722, and 0.26 points for adaptive multirate wideband codec(AMR-WB). In a subjective CCR listening test, the proposed postprocessor on G.711-coded speech exceeds the speech quality of an ITU-T standardized postfilter by 0.36 CMOS points, and obtains a clear preference of 1.77 CMOS points compared to G.711, even en par with uncoded speech.

**Index Terms—convolutional neural networks, speech codecs, speech enhancement.**

If you use **Convolutional Neural Networks to Enhance Coded Speech** in your research, please cite:
```bibtex
@article{cnn2codedspeech,
  title={Convolutional Neural Networks to Enhance Coded Speech},
  author={Zhao, Ziyue and Liu, Huijun and Fingscheidt, Tim},
  journal={Transactions on Audio, Speech and Language Processing},
  year={2018}
}
```


