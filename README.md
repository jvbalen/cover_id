## Differentiable Fingerprinting: a convolutional architecture for cover song detection

Work in progress on 'learning to fingerprint' for challening audio-based content ID problems, such as cover song detection.

Currently focused on experiments in which a fingerprint is learned from a dataset of cover songs. The main idea behind this is explained in our [Audio Bigrams](http://dspace.library.uu.nl/handle/1874/314940) paper [1].

Very briefly explained:
1. most fingerprints encode some kind of co-occurrence of salient events  
 (e.g., Shazam's landmark-based fingerprinter, 'intervalgrams'...)

1. 'salient event detection' can be implemented as a convolution:  
 `conv2d(X, W)`  
 with W the 'salient events'.

1. co-occurrence can be implemented as  
 `conv2d(X, w) @ X.T`  
 with w a window and @ the matrix product.

1. all of this is differentiable, therefore, any fingerprinting system that can be formulated like this can be trained 'end-to-end'.

To evaluate the learned fingerprint, we compare to the elegant and performant '2D Fourier Transform Magniture Coeffients' by Bertin-Mahieux and Ellis [2], and a simpler fingerprinting approach by Kim et al [3].

We use the [Second-hand Song Dataset](http://labrosa.ee.columbia.edu/millionsong/secondhand) with dublicates removed as proposed by [Julien Osmalskyj](http://www.montefiore.ulg.ac.be/~josmalskyj/code.php).

[1] Van Balen, J., Wiering, F., & Veltkamp, R. (2015). [Audio Bigrams as a Unifying Model of Pitch-based Song Description](http://dspace.library.uu.nl/handle/1874/314940).

[2] Bertin-Mahieux, T., & Ellis, D. P. W. (2012). [Large-Scale Cover Song Recognition Using The 2d Fourier Transform Magnitude](http://academiccommons.columbia.edu/download/fedora_content/download/ac:159481/CONTENT/BertE12-2DFTM.pdf). In Proc. International Society for Music Information Retrieval Conference.

[3] Kim, S., Unal, E., & Narayanan, S. (2008). [Music fingerprint extraction for classical music cover song identification](http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=4607671&url=http%3A%2F%2Fieeexplore.ieee.org%2Fxpls%2Fabs_all.jsp%3Farnumber%3D4607671). IEEE Conference on Multimedia and Expo.

More information in [this notebook](./learn.ipynb).

---

(c) 2016 Jan Van Balen

[github.com/jvbalen](www.github.com/jvbalen) - [twitter.com/jvanbalen](www.twitter.com/jvanbalen)
