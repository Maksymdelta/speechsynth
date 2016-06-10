# speechsynth
**I gave up.**

An incomplete speech analysis/synthesis tool library implement in Python

# Files

### F0
* pyin.py: Probabilistic YIN F0 Estimator.
* MonoPitch.py: HMM F0 tracker for pyin.

### Basic speech model
* hnm.py: Harmonic Noise Model(Support QFFT and QHM-AIR).
* qhm_slove.py: Quasi-Harmonic Model Adaptive Iterative Refinement(QHM-AIR).

### Formant & Spectrum Envelope
* correcter.py: Formant trajectory correcter.
* envelope.py: Spectrum envelopes(TrueEnvelope, MFIEnvelope).
* fecsola.py: Klatt Combination FECSOLA(Spectrum formant transform).
* formantMarker.py: Formant marking tool (GUI).
* lpc.py: Linear Predictive Coding(Support Autocorrelation and Burg).
* mfcc.py: Mel-frequency cepstral coefficients(MFCC) generator.

### Voicebank
* synther.py: Synth a Track.
* voicedb.py: Voicebank Database.
* dbsuite.py: Voicebank creating helper functions(include some GUI classes).
* note.py: Note, Track, Project.
* tune.py: pitch, freq.

### Common
* common.py: Common functions.
* SparseHMM.py: Sparse Hidden Markov Model Viterbi decoder.
* psola.py: (incomplete). Glottal Pulse Extractor.
* stft.py: F0 Adaptive Short-time Fourier Transform
* utils.py: Similar to commons.py. Maybe used in some script.
* match.py: Match Data Structure.(Used in note.py->Track)

### Others
* test.py: test script.

# Works cited
* De Cheveigné A, Kawahara H. YIN, a fundamental frequency estimator for speech and music[J]. The Journal of the Acoustical Society of America, 2002, 111(4): 1917-1930.

* Mauch M, Dixon S. pYIN: A fundamental frequency estimator using probabilistic threshold distributions[C]//Acoustics, Speech and Signal Processing (ICASSP), 2014 IEEE International Conference on. IEEE, 2014: 659-663.

* Serra, Xavier. "A system for sound analysis/transformation/synthesis based on a deterministic plus stochastic decomposition." Diss. Universitat Pompeu Fabra. 1989.

* Pantazis Y, Stylianou Y. Improving the modeling of the noise part in the harmonic plus noise model of speech[C]//Acoustics, Speech and Signal Processing, 2008. ICASSP 2008. IEEE International Conference on. IEEE, 2008: 4609-4612.

* Pantazis Y, Tzedakis G, Rosec O, et al. Analysis/synthesis of speech based on an adaptive quasi-harmonic plus noise model[C]//Acoustics Speech and Signal Processing (ICASSP), 2010 IEEE International Conference on. IEEE, 2010: 4246-4249.

* Degottex G, Stylianou Y. Analysis and synthesis of speech using an adaptive full-band harmonic model[J]. Audio, Speech, and Language Processing, IEEE Transactions on, 2013, 21(10): 2085-2095.

* Röbel, Axel, and Xavier Rodet. "Efficient spectral envelope estimation and its application to pitch shifting and envelope preservation." Proc. DAFx. 2005.

* Nakano T, Goto M. A spectral envelope estimation method based on f0-adaptive multi-frame integration analysis[C]//SAPA-SCALE Conference. 2012.

* https://github.com/Sleepwalking/CVEDSP

* Gray Jr A H, Wong D Y. The Burg algorithm for LPC Speech analysis/synthesis[J]. Acoustics, Speech and Signal Processing, IEEE Transactions on, 1980, 28(6): 609-615.

* Moulines, Eric, and Francis Charpentier. “Pitch-synchronous waveform processing techniques for textto-speech synthesis using diphones.” Speech communication 9.5 (1990): 453-467.

* http://www.fon.hum.uva.nl/praat/
