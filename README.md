## eeg-classification
Train a classifier of human activity on EEG data

### notes and such

- Doing an SVD on all is a bad idea
- do a CSP to reduce the number of channels (the electrodes) into an X amount of "meta"-electrodes (see [this](https://doc.ml.tu-berlin.de/bbci/publications/BlaTomLemKawMue08.pdf))
- then it's super confusing as in how many models are needed and for what
- also the subject isn't clear yet
