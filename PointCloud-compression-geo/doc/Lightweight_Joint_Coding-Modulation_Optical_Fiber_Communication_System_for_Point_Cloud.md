# Lightweight Joint Coding-Modulation Optical Fiber Communication System for Point Cloud

**Wei Zhang, Zhenming Yu, Member, IEEE, Hongyu Huang, Xiangyong Dong, Kaixuan Sun, and Kun Xu**

---

## Abstract

Achieving efficient point cloud (PC) transmission is a fundamental requirement for immersive holographic-type communication. However, traditional optical fiber communication (TOFC), based on separated coding modulation for PC transmission, faces challenges related to massive data transmission and heavy computational resource requirements. To achieve lightweight and efficient PC transmission, we propose and experimentally demonstrate a joint coding-modulation optical fiber communication system for PC transmission (JCMPC). A joint encoding-modulation (JEM) network based on 3D convolution is designed to encode the PC into symbols for transmission directly. At the receiver, a joint decoding-demodulation (JDD) network is used to reconstruct the signals transmitted through the communication channel into the received PC. The experimental results indicate that the proposed JCMPC outperforms transmission schemes based on separate coding modulation and exhibits gradual performance degradation with the deterioration of channel conditions. We evaluate the decoding computational complexity of our proposed JCMPC scheme against the separate transmission schemes using Geometry-based Point Cloud Compression (G-PCC) and Low-Density Parity-Check (LDPC) codes. The results demonstrate that JCMPC reduces the decoding computational operations by over 80% compared to TOFC.

**Index Terms**— Deep learning, optical fiber communication, joint coding-modulation, point cloud compression.

## I. INTRODUCTION

POINT cloud (PC) refers to a collection of discrete points that represent the spatial positions and characteristics (e.g., color, reflectance, and albedo) in a three-dimensional (3D) space [1]. Tbps-level bandwidth and end-to-end latency of less than 20ms are needed to meet the requirements for holographic-type communication [2], [3]. Optical fiber communication, with its advantages of high speed and high capacity, has the potential to achieve efficient PC transmission. Shanthi et al. achieved a holographic-type communication system implementation for remote communication with a latency of around 131ms through 100Gbps optical links [4]. However, there are still some challenges related to the cliff effect and heavy computational resource requirements to achieve efficient PC transmission in traditional optical fiber communication (TOFC) based on separated coding modulation.

TOFC employs a separate design in the transmission of PC: source coding, channel coding, and modulation [5]. For source coding of PC, there are two main categories: those based on traditional models [1], [6]–[8] and those based on deep learning [9]–[17]. The classic approach for the first category is Geometry-based Point Cloud Compression (G-PCC) proposed by Moving Picture Expert Group (MPEG) [1]. However, using the MPEG standard for PC compression demands substantial computational resources. Heavy computation impacts the user experience of devices with limited computing resources. Recently, the second category has shown better performance than G-PCC in the field of source coding for PC. For channel coding, schemes such as Low-Density Parity-Check code (LDPC) [18] have proved to be close to the Shannon limit. The bits encoded by channel coding are mapped to symbols by modulation (e.g., PAM4, PAM8).

The Shannon separation theorem proves that this source-channel separation coding method can achieve theoretical optimality [19]. However, emerging applications like holographic communication and virtual reality (VR) require efficient transmission of PCs with constraints on low latency, low complexity, and limited bandwidth. These constraints make it challenging to meet the assumption of infinitely long source and channel blocks in the separation theorem [19]–[21]. In addition, separation-based communication systems may face a cliff effect [22], [23]. The cliff effect has two characteristics. First, when the system's communication rate does not match the current channel conditions, its performance may degrade severely. Second, once the system's rate is fixed, even under favorable channel conditions, the reconstruction quality cannot be improved. Recently, deep learning-based joint source-channel coding (JSCC) schemes have attracted significant interest in communications [24]–[27]. The deep learning-based JSCC schemes can intelligently analyze and extract source characteristics from large amounts of training data, directly mapping the source to discrete-time analog symbols for transmission. This novel paradigm demonstrates higher compression capabilities and greater robustness to channel variations.

To achieve high-capacity lightweight PC transmission and overcome the cliff effect, we propose and experimentally demonstrate a joint coding-modulation optical fiber communication system for PC transmission (JCMPC). The designed proposed JCMPC framework in this work offers a novel approach to PC transmission and significantly reduces the computational complexity. The goal of the JCMPC is no longer bits error-free transmission but to deliver the desired information that is guided by the transmission task. We propose a 3D convolution-based joint encoding-modulation (JEM) network to extract the features of PC and jointly encode-modulate them into symbols for transmission directly. This makes it possible for JCMPC to achieve higher PC transmission capabilities and more stable performance under limited bandwidth. We achieve a lightweight network design of JCMPC by using a shallow network structure and small convolutional kernels, thereby reducing the system's computational complexity. Additionally, we design a deep learning-based PC source coding scheme (DeepPC). Unlike JCMPC, the transmission scheme based on DeepPC adopts a separate design (i.e., DeepPC+LDPC+PAM4). DeepPC adopts the same network architecture as JCMPC to ensure that they have similar point cloud compression capabilities. By comparing JCMPC to DeepPC, we can demonstrate that the performance improvement of JCMPC in PC transmission comes not only from the neural network but also from the joint coding-modulation architecture.

Both numerical simulations and optical transmission experiments are carried out to analyze the performance of JCMPC. We conduct numerical simulations under additive white Gaussian noise (AWGN) channel conditions. Numerical simulation results show that under various compression ratios (CRs) and Signal-to-noise Ratio (SNR) conditions, JCMPC outperforms transmission schemes based on separate coding modulation in terms of point-to-point peak signal-to-noise ratio (D1-PSNR) and point-to-plane peak signal-to-noise ratio (D2-PSNR) and visual results for point cloud transmission quality. Moreover, the proposed system exhibits gradual performance degradation with a decrease in SNR, while the other schemes suffer a cliff effect. Then, we conduct experiments on an intensity modulation/direct detection (IM/DD) optical fiber link to verify the performance of JCMPC in real optical fiber communication systems. JCMPC exceeds the transmission schemes based on G-PCC+LDPC by more than 12dB in terms of D1-PSNR and D2-PSNR across various CRs. Besides, JCMPC is more robust to optical link impairments. Using D1-PSNR of 62dB as a reference, the receiver sensitivity of JCMPC is improved by at least 2dB compared to G-PCC+LDPC. JCMPC can transmit over longer distances without significant loss of PC reconstruction quality. Moreover, benefiting from the joint coding-modulation structure and lightweight network, our proposed approach reduces decoding computation by more than 80% compared to traditional schemes.

## II. SYSTEM ARCHITECTURE

In this section, we first provide a detailed discussion of the architecture of our proposed JCMPC, which includes preprocessing, JEM, joint decoding-demodulation network (JDD), and postprocessing. Then, we present the system framework of DeepPC, which is a source coding scheme for PC. Unlike JCMPC, DeepPC introduces a quantization and entropy model to convert symbols into bitstreams. Finally, we introduce the settings of the loss functions and optimization strategies for both schemes.

### A. Overview of the JCMPC

The framework of JCMPC is illustrated in Fig. 1. Preprocessing involves two steps. First, the PC is voxelized, which is advantageous for extracting feature information from PCs through 3D convolutions. Second, the voxelized PC is divided into many equal-sized cubes. The JEM analyzes each cube, capturing spatial redundancies and encoding the cubes into discrete-time analog symbols. The JDD reconstructs the symbols after forward feedback equalization (FFE) into voxelized cubes. Finally, in post-processing, the original coordinates are restored, and then the cubes are merged to obtain the final fully reconstructed PC.

### B. Point Cloud Preprocessing

As shown in Fig. 1(a), the preprocessing of the JCMPC consists of two steps: voxelization and partition. PC geometry consists of numerous unordered 3D point coordinates $(i,j,k)$. Unlike images with ordered pixels, PCs usually exhibit irregularity. For irregular PCs, obtaining correlations between 3D points is challenging. Therefore, the JCMPC adopts a voxelized PC representation structure.

**Voxelization** is the process of representing PC geometry using a uniformly sized voxel grid in space [1]. The 3D space is discretized into neatly arranged voxel cells, with occupied voxels labeled as 1 and empty voxels labeled as 0. The depth of voxelization provides the spatial resolution of the PC. A PC voxelized over a $2^n \times 2^n \times 2^n$ grid is called an n-depth PC. For example, a cube of $1024 \times 1024 \times 1024$ voxels is known as depth 10. This representation can create regular 3D blocks, which are advantageous for extracting feature information from PCs through 3D convolutions.

**Partition**: PCs typically contain a substantial amount of data, which makes it difficult to encode an entire PC at once. Taking inspiration from the successful application of block-based processing in image processing, we partition the voxelized PC into numerous equally sized cubes. The size of each cube is set to $32 \times 32 \times 32$. Because each cube is independent without considering correlations between cubes, the JEM can process multiple cubes in parallel. This approach reduces both time consumption and resource utilization.

### C. Joint Encoding-Modulation Network

In traditional PC transmission systems, three separate steps are required to generate the symbols transmitted through the channel. First, source coding removes source redundancy to achieve compression. Then, channel coding adds correction codes to counteract noise in the communication channel. Finally, modulation converts the bits into symbols for transmission. JCMPC bypasses the bit-based coding and modulation, instead using JEM to extract the features of the input PC cubes $x$ and directly map them into symbols $y$ for transmission. These channel symbols $y$ exhibit discrete-time and continuous amplitude characteristics. The constellation of $y$ is irregular, offering greater freedom compared to the discrete constellations of TOFC.

In recent years, CNNs have been widely used in image-related tasks, such as image compression, image classification, and object detection, showing great potential [28]–[30]. Thus, we propose a 3D convolution-based JEM for PC transmission. Fig. 1(b) illustrates the JEM framework. The size of input PC cube $x$ is $C \times W \times W \times W$, where $C = 1$ denotes the occupancy of voxels and the spatial resolution $W$ of PC cube is 32. The JEM maps the input PC cube $x$ to symbols $y$ with the size of $N \times 2 \times 2 \times 2$, where $N$ is the number of filters of 3D convolution layers. The $y$ is then reshaped to one-dimensional symbols for transmission in the communication channel. The JEM primarily consists of 3D convolutional layers, with two residual blocks (ResBlocks) serving as the fundamental unit. The structure of the ResBlock is illustrated in Fig. 2. It comprises three convolutional layers with kernel sizes of $(3,3,3)$ and $(5,5,5)$. Using larger convolutional kernels can increase the receptive field but significantly increase the number of parameters. Therefore, in our work, we combine convolutional kernels of different sizes to enlarge the receptive field while maintaining a low complexity. To enhance the fusion of deep-level and shallow-level features and alleviate the vanishing gradient problem, a skip connection is introduced.

### D. Joint Decoding-Demodulation Network

As shown in Fig. 1(c), JDD is the inverse process of JEM. Taking the signals after channel $\hat{y}$ as input, the JDD ultimately produces the decoding output $\bar{x}$ of size $1 \times 32 \times 32 \times 32$, where the first channel represents the probability of voxel occupation. In fact, the PC reconstruction process can be viewed as a classification problem, with the goal of classifying each voxel as empty or occupied. The Sigmoid activation function is commonly used in classification tasks. Therefore, from the choice of activation functions, the final output layer of the JDD adopts the Sigmoid activation function to obtain the probability of voxel occupation. The probability is represented by a floating point number between 0 and 1. Through the postprocessing, the probability is judged to 0 or 1 to indicate whether the voxel is occupied, where 0 indicates that the voxel is empty and 1 indicates that the voxel is occupied.

### E. Point Cloud Postprocessing

Once all the cubes have been reconstructed, a postprocessing procedure is employed to restore the voxel representation back to the original coordinates and merge the cubes into a complete PC (Fig. 1(d)). It should be noted that the coordinates of each voxel in the output of the JDD are floating point numbers distributed between 0 and 1. Therefore, the quantization with a fixed threshold value of $t = 0.5$ is performed to classify them as either 0 or 1. Finally, all cubes are merged into the restored complete PC.

### F. DeepPC Model

To further explore the improvements brought about by the joint coding-modulation architecture, we design a DeepPC, which is a source coding scheme for PC. The dashed box in Fig. 3(b) illustrates the structure of the DeepPC. To achieve similar PC compression capabilities between DeepPC and JCMPC, the network structures of the encoder and decoder for Deep-PC are identical to those of JEM and JDD, respectively. DeepPC also employs the same preprocessing and postprocessing as JCMPC. Unlike JCMPC, which is based on joint coding-modulation (Fig. 3(a)), the transmission scheme using DeepPC is based on separate coding modulation.

Specifically, the encoder of DeepPC first encodes the input PC cubes into latent representation $y$. Subsequently, this latent representation $y$ is quantized to $\hat{y}$. The quantized representation $\hat{y}$ is then encoded into a bitstream using entropy coding. Following this, a channel encoder is applied to protect it against the impairments introduced by the communication channel. Finally, the coded bitstream is mapped to transmission symbols $z$ through modulation. In this work, the modulation formats used are PAM4 and PAM8. For the receiving end, the symbols transmitted through the channel $z$ are transformed into a bitstream by demodulation and channel decoding. The bitstream is then converted into the received PC through entropy decoding and the decoder of the DeepPC.

Independent uniform noise is approximated to the quantization error and is frequently utilized as a model of the quantization error [31]. Due to the non-differentiable nature of quantization during training, the quantization is replaced by additive uniform noise to ensure the differentiability for the back-propagation operation:

$$\tilde{y} = y + u$$

where $u$ is random uniform noise ranging from $-\frac{1}{2}$ and $\frac{1}{2}$.

During the testing phase, the latent representations $y$ are quantized to $\hat{y}$. $\hat{y}$ represents latent representations with actual rounding error.

### G. Model Training

To enhance the robustness of the model to noise, JCMPC is trained using an end-to-end approach in the AWGN channel, and a random SNR strategy with a range of 4–20 dB is adopted in the training. Gaussian noise is added to the output symbols from the JEM, and the noisy symbols are then fed into the JDD for further processing.

In voxel-based PC, 0 or 1 can be used to represent whether a voxel is occupied, so the reconstruction process can be viewed as a binary classification problem. Due to the nature of PC, the number of 0 values that represent empty space is usually much larger than the 1 values that represent space occupation. In our work, we adopt the focal loss as the training loss function to address the issue of imbalanced distribution between occupied and unoccupied samples in classification tasks [32]. The formula is as follows:

$$FL(p_{\bar{x}}) = \frac{1}{N_V}\sum_{N_V}[-\alpha(1-p_{\bar{x}})^{\gamma}\log(p_{\bar{x}}) - (1-\alpha)p_{\bar{x}}^{\gamma}\log(1-p_{\bar{x}})]$$

where $p_{\bar{x}} = \text{sigmoid}(x)$ represents the output value of the decoded voxel, ranging from 0 to 1, which represents the probability of voxel occupancy. $N_V$ denotes the total number of voxels. To address the issue of data imbalance, a weight factor $\alpha$ is introduced to adjust the weights of different samples. The best $\alpha$ value depends on the class imbalance level of the PC. A larger $\alpha$ increases the importance of voxels with a 1 value. Sparser PCs typically prefer higher $\alpha$ values so that originally filled voxels are not wrongly discarded. In our experiments, we set $\alpha$ to 0.95.

The training of DeepPC model also follows an end-to-end manner. Unlike JCMPC, the channel coding and modulation are not jointly optimized in the DeepPC. A loss function is employed to maximize the overall rate-distortion performance of DeepPC. The loss function of the model consists of a distortion term $D$, rate term $R$, and a weight term $\lambda$, i.e.,

$$\text{Loss} = R + \lambda D$$

where the rate term $R$ is the bit rate of quantized latent representations, and the distortion term $D$ applies the same focal loss as in (4). We can approximate the actual bit rate via entropy of the quantized latent representation [33], [34]:

$$R = -E[\log_2 p_{\hat{y}}(\hat{y})]$$

where $p_{\hat{y}}(\hat{y})$ is the self-probability density function of $\hat{y}$. By adjusting the $\lambda$, we can tune the trade-off between bit rate and transmission quality. In other words, by setting a higher weight, the model will focus more on learning how to preserve geometric information rather than compressing, leading to higher transmission quality at the expense of a higher bit rate.

We analyzed the impact of different $\lambda$ values on the performance of DeepPC, using the Phil PC for testing. As shown in Fig. 4, both the D1-PSNR and D2-PSNR initially increase and then decrease as $\lambda$ increases. The optimal $\lambda$ value is 20, so we ultimately selected $\lambda$ to be 20 in our work.

## III. NUMERICAL SIMULATION

To compare the performance of the proposed JCMPC with transmission schemes based on separate coding modulation, extensive numerical simulations are conducted on AWGN channels. The ideal channel condition that can achieve Shannon capacity is also considered to simulate the performance upper limit of the schemes. Specifically, we use the Gaussian capacity formula [5] to calculate the maximum transmission rate for the ideal DeepPC+Capacity and G-PCC+Capacity. Additionally, we evaluate the decoding complexity of JCMPC and classical PC transmission schemes.

### A. Simulation Setup

**1) Datasets**: We select 38 PCs from various repositories to compose the dataset for JCMPC and DeepPC. The 38 voxelized PCs are divided into a total of 49,197 cubes with a size of $32 \times 32 \times 32$ as the training dataset of the network. This dataset includes PCs from libraries such as 8iVFB, JPEG Pleno, PointXR, and VESENSE [35]-[38]. The 8iVFB datasets include four dynamic PC sequences of human bodies, each comprising multiple frames of the same individual. The JPEG Pleno datasets contain real and synthetic PCs representing cultural heritage sites and the human body. The PointXR dataset consists of high-quality PCs representing cultural heritage models. The VSENSE dataset consists of two dynamic sequences of human bodies. Samples of the PCs used for training are illustrated in Fig. 5(a). To evaluate the performance of our proposed framework, tests were conducted on two PCs, as shown in Fig. 5(b).

**2) Comparison Schemes**: We compare JCMPC with schemes based on separate coding modulation. The source coding includes both the popular PC compression scheme G-PCC proposed by MPEG and DeepPC. The geometry model of the G-PCC is an octree. The compression rate of G-PCC is controlled by adjusting the positionQuantizationScale parameter. For channel coding, a 3/4-rate LDPC code is used. The modulation formats applied are PAM4 and PAM8. Additionally, an ideal channel coding and modulation scheme that can reach Shannon capacity is included in the comparison. For example, the scheme using G-PCC combined with ideal capacity-achieving channel coding and modulation (marked as G-PCC+Capacity) can be seen as the upper limit of performance when using G-PCC as the source coding. To ensure fairness of comparison, the CRs of all schemes are controlled to be the same.

**3) Evaluation Metrics**: We use classical D1-PSNR and D2-PSNR [39] to evaluate the transmission quality of the PC. D1-PSNR and D2-PSNR measure the geometric distortions of PC in terms of point-to-point distance and point-to-plane distance, respectively. Let A and B denote the original and reconstructed PC, respectively.

For D1-PSNR, the mean squared error $e_{A,B}^{D1}$ can be represented as follows:

$$e_{A,B}^{D1} = \frac{1}{N_A}\sum_{a_i \in A} \|a_i - b_k\|_2^2$$

where $a_i$ is a point in PC A, $b_k$ is its nearest neighbor in PC B, and $N_A$ is the number of points in PC A.

For D2-PSNR, the mean squared error $e_{A,B}^{D2}$ can be represented as follows:

$$e_{A,B}^{D2} = \frac{1}{N_A}\sum_{a_i \in A} [(a_i - b_k) \cdot n_i]^2$$

where $a_i$ is a point in PC A, $b_k$ is its nearest neighbor in PC B, $n_i$ is the normal vector on point $a_i$ in PC A, and $N_A$ is the number of points in PC A.

Then, the PSNR is calculated as follows:

$$\text{PSNR}_{A,B} = 10\log_{10}\frac{p^2}{e_{A,B}^{Di}}$$

where $i = 1$ and $2$ are used for D1-PSNR and D2-PSNR, respectively. The $p$ is the maximum value of the distances for all points $a_i$ in the input PC.

### B. Performance Evaluation

**Compression Ratio Performance**: Fig. 6 shows the results of various transmission schemes under different CRs at the AWGN channel with SNR = 16 dB, where the distortion is measured in terms of D1-PSNR (point-to-point) and D2-PSNR (point-to-plane). The tested PCs are Phil and Longdress from MPEG. The CR is formulated as follows:

$$\text{CR} = \frac{N_s}{N_p}$$

where $N_s$ is the number of symbols for transmission and $N_p$ is the number of points of the input PC. From the results, we can see that the JCMPC proposed by us outperforms the schemes based on G-PCC for all CRs. Compared to the G-PCC+capacity under ideal conditions, the proposed JCMPC still shows excellent performance and achieves more than a 10 dB improvement in D1-PSNR for all two datasets. Furthermore, DeepPC+LDPC can outperform G-PCC+LDPC, which means that DeepPC has better PC compression ability than G-PCC. It is worth noting that the performance of JCMPC is better than DeepPC+LDPC, which indicates that the performance improvement of JCMPC comes not only from the excellent PC compression capability of the neural network but also from the improvement brought by the architecture of joint coding-modulation. In addition, the proposed JCMPC can even surpass DeepPC+Capacity, which is an upper bound of DeepPC combined with channel coding and modulation. However, this does not imply that JCMPC can transcend the Shannon capacity, as DeepPC suffers performance losses due to the quantization of y and cannot represent the upper limit of JCMPC.

**SNR Performance**: Fig. 7 demonstrates the D1-PSNR and D2-PSNR performance with the change in SNR, where the average CR is 0.374 for Phil and 0.41 for Longdress. The PC transmission rates of all schemes are controlled at the same. When the SNR is greater than 16 dB, using G-PCC compression on the Longdress PC under ideal capacity-achieving conditions can support lossless PC transmission. In other cases, for both test PCs, JCMPC generally outperforms other transmission schemes. When the testing SNR decreases, the cliff effect appears in G-PCC+LDPC and DeepPC+LDPC, which are based on separated coding modulation, while the proposed JCMPC achieves smooth degradation. Specifically, for transmission schemes based on separate coding modulation, once the coding and modulation schemes are selected, the quality of the reconstructed PC does not improve with the increase of SNR. Additionally, as channel conditions deteriorate, the performance experiences a cliff-like degradation due to reliance on entropy coding and channel coding. Unlike bit-based transmission schemes, JCMPC exhibits a smooth performance decline with decreasing SNR, which avoids the cliff effect. DeepPC+LDPC also involve the cliff effect due to the use of entropy coding and LDPC.

### C. Visual Assessment

We present the visual results of different transmission schemes, as depicted in Fig. 8, at SNR = 16dB. To provide a more intuitive visualization of the PC geometry, we refer to the approach in [16]. We initially calculate normals for each point, using a set of 100 neighboring points, and then we set parallel lighting in the front and render the points. It is evident that our proposed JCMPC accurately preserves the original PC details while producing smoother reconstructions. The G-PCC+LDPC scheme loses a significant amount of detail and is considerably sparser than the ground truth when compared to the JCMPC. It is worth noting that the PCs transmitted by our proposed method fill in the broken parts of the ground truth, which may have occurred during the scanning process. For example, the details of the man's arm in Fig. 8 are significantly improved. We attribute this improvement to the ability of JCMPC to learn geometric features from a large amount of high-quality training data. This capability enables it to reconstruct the PC more completely and smoothly, which is a characteristic not present in traditional methods.

Based on the point-to-point Hausdorff distance between the reconstructed PC and the ground truth PC, we generate an error map, as depicted in Fig. 9. The color in the figure represents the point-to-point distances between the reconstructed PC and the ground truth PC. The error map demonstrates that the reconstructed PC from our proposed approach is more closely aligned with the original PC. The main source of distortions in JCMPC and DeepPC+LDPC can be attributed to the seams generated by block-based processing, while the distortions in G-PCC+LDPC are random.

### D. Computational Complexity

G-PCC and LDPC are known to have high computational complexities, especially during the decoding process. While it has become feasible to transmit text, videos, and other content on mobile devices like smartphones, delivering large-scale data like PCs in real-time on resource-constrained devices remains a significant challenge [40]. We measured the computational complexity of JCMPC and G-PCC+LDPC by counting the number of computational operations during the decoding process, including both multiplication and addition operations. For JCMPC, symbols after equalization are converted to decoded PC through the JDD and postprocessing. For G-PCC+LDPC, the decoding process includes demodulation, channel decoding, and source decoding. The demodulation formats applied are PAM4 and PAM8, the channel decoding employs a rate-3/4, block length-1944 LDPC with 50 iterations, and the source decoding uses G-PCC based on octree. Since the complexity of demodulation is much lower compared to channel decoding and source decoding, we ignore it in our calculations.

The tested PC used the Longdress dataset from 8iVFB. Since the CR of each transmission scheme is the same, a higher modulation order requires adjusting the code rate of the source coding to generate more bits. Therefore, the decoding operations of the transmission scheme using PAM8 are approximately 1.5 times that of the PAM4 scheme. Benefiting from the joint coding-modulation architecture and the lightweight network structure, JCMPC significantly reduces computational complexity. As shown in Fig. 10, decoding Longdress with JCMPC requires only 1.58G operations, while G-PCC+LDPC+PAM4 and G-PCC+LDPC+PAM8 require 9.39G and 14.75G operations, respectively. Compared to TOFC, JCMPC reduces computational operations by at least 83.17%. The significant reduction in decoding computational load further facilitates the real-time reception of large-scale PC data on devices with limited computing resources.

## IV. EXPERIMENTAL SETUP AND RESULTS

In optical fiber communication systems, the performance of transmission schemes can be affected by optical link impairments. To validate the performance of JCMPC in practical fiber optic communication systems, we conducted experiments using the IM/DD-based optical fiber link.

### A. Experimental Setup

To evaluate the performance of JCMPC in optical fiber communication systems, the experimental setup of the optical transmission system is shown in Fig. 11. The digital signal processing at the transmitter (Tx-DSP) includes the square root raised cosine (SRRC) filtering and resample. The symbols generated by the JEM are shaped using the SRRC with a 0.1 roll-off factor and then resampled to 65 GSa/s. The signals are then sent into an arbitrary waveform generator (AWG, Keysight M8195A). This AWG offers a maximum sampling rate of 65 GSa/s and an analog bandwidth of 25 GHz. A linear broadband amplifier (SHF S807C) amplifies the electrical analog signals generated by the AWG, which are then converted into optical signals using a Mach-Zehnder modulator (MZM) (EOSPACE AX-0MVS-40-PFA-PFA-LV). The laser source (CoBrite-DX) launches an optical wave at 1550nm with an optical power of 10 dBm and feeds it to the MZM. In the optical back-to-back (OB2B) transmission experiment, the received optical power (ROP) is controlled by a variable optical attenuator (VOA). For the optical transmission experiment, both the input fiber optical power and the ROP are maintained at 0 dBm by employing VOA and an erbium-doped fiber amplifier (EDFA). At the receiver, the optical signals are converted back into electrical signals using a photodetector (PD, FINISAR XPRV2325A) with a 30-GHz bandwidth. The received electrical signals are sampled at a rate of 100 GSa/s using a digital sampling oscilloscope (DSO, Tektronix DSA72504D) with an analog bandwidth of 25 GHz. These electrical signals are processed by the digital signal processing at the receiver (Rx-DSP), which includes resample, matched filtering, and a 61-tap FFE both for JCMPC and comparison schemes.

For the source encoding of the comparison scheme, we employ an MPEG G-PCC, based on an octree structure to compress the PC into bits. For the channel coding of TOFC, a 3/4-rate LDPC code is used. The modulation formats applied are PAM4 and PAM8. The symbol and sampling rates are set at 34 Gbaud and 60 GSa/s, respectively. The PC transmission rates of all transmission schemes are controlled to be the same.

### B. Performance Evaluation

**Back-to-Back Transmission**: Fig. 12 shows the D1-PSNR and D2-PSNR results versus the change of CRs over the OB2B transmission system at ROP = 0 dBm. The tested PC used the Phil from MPEG. Experimental results demonstrate that the JCMPC consistently outperforms G-PCC+LDPC at various CRs. Compared to the PAM8 transmission, the JCMPC achieves an average improvement of 12.61 dB in D1-PSNR and 12.51 dB in D2-PSNR. Compared to the PAM4 transmission, the JCMPC improves by an average of 14.47 dB in the D1-PSNR and 13.88 dB in the D2-PSNR.

**Received Optical Power Performance**: The results of the experiments conducted in an OB2B transmission system under different ROPs are shown in Fig. 13. The tested PC is the Phil from the MPEG. The average CR is set to 0.374. The PC transmission rates of JCMPC and TOFC are controlled at the same, about $2.6 \times 10^5$ PCs/s. The experimental results indicate that the TOFC, based on separate coding modulation, experiences cliff effects at ROP levels of -8 dBm for PAM4 and -3 dBm for PAM8. TOFC needs to utilize low-order modulation in the low ROP region. When adopting low-order modulation, source coding (G-PCC) needs to employ higher CRs to maintain the same PC transmission rate as the system. Higher CRs lead to increased PC distortion. In contrast, JCMPC shows stable performance degradation with decreasing ROP. By setting a 62 dB D1-PSNR as the reference, the receiver sensitivity in the JCMPC is at least 2 dB higher than the transmission scheme based on separate coding modulation. These results demonstrate that JCMPC can adapt to worse channel conditions and exhibits stronger robustness to optical link impairments.

**Transmission Distance Performance**: Fig. 14 shows the D1-PSNR and D2-PSNR performance versus the different transmission distances while ensuring that the input optical power and ROP are both set to 0 dBm. The average CR is set to 0.374. The tested PC is the Phil from the MPEG. As the transmission distance increases, chromatic dispersion (CD) seriously degrades signal quality [41]. Without special CD compensation, TOFC can reliably transmit PCs within a range of only 10 km. In contrast, JCMPC can transmit PCs over distances exceeding 80 km without significant degradation in quality. This significant improvement in the achievable transmission distance highlights the capability of the JCMPC to maintain reliable long-distance transmission.

### C. Visual Assessment

To facilitate a more intuitive comparison of transmission performance across different transmission schemes, we present visual results, as depicted in Fig. 15 and Fig. 16, in an OB2B transmission system with ROP set at 0 dBm. The visualization setup is the same as in Fig. 8. From the results, it can be observed that under the same channel conditions and CR, our proposed JCMPC achieves higher visual quality without losing too many details like TOFC. An error map based on the point-to-point Hausdorff distance between the received PC and the ground truth PC is depicted in Fig. 17. Similar to the numerical simulation results, the reconstructed PC from our proposed approach is more closely aligned with the original PC. Especially for smoother PCs like Longdress, JCMPC can achieve more accurate reconstruction.

## V. CONCLUSION

In this paper, we propose and experimentally demonstrate an efficient and lightweight PC optical transmission scheme based on joint coding-modulation. Unlike the traditional bit-based separate coding modulation scheme, JCMPC maps the input PC to symbols for transmission directly. By adopting a novel joint coding-modulation architecture and lightweight network structure, JCMPC reduces the decoding computational operations by over 80% compared to TOFC. Both numerical simulations and optical transmission experiments show that JCMPC outperforms separate coding modulation schemes across various CRs, ROPs, and transmission distances. The proposed JCMPC can achieve an improvement of more than 10 dB in both D1-PSNR and D2-PSNR compared to G-PCC+LDPC and even surpasses the DeepPC with ideal channel conditions. Additionally, JCMPC improves receiver sensitivity by at least 2 dB and can achieve longer distance PC transmission. In summary, this paper proposes a method for achieving high-capacity and low-complexity PC transmission in optical fiber communication systems.

## REFERENCES

[1] S. Schwarz et al., "Emerging MPEG Standards for Point Cloud Compression," IEEE J. Emerg. Sel. Top. Circuits Syst., vol. 9, no. 1, pp. 133-148, Mar. 2019, doi: 10.1109/JETCAS.2018.2885981.

[2] A. Clemm, M. T. Vega, H. K. Ravuri, T. Wauters, and F. D. Turck, "Toward Truly Immersive Holographic-Type Communication: Challenges and Solutions," IEEE Commun. Mag., vol. 58, no. 1, pp. 93-99, 2020.

[3] Ian F. Akyildiz and Hongzhi Guo, "Holographic-type communication: A new challenge for the next decade," ITU J-FET, vol. 3, Issue 2, pp. 421-442, Sep. 2022.

[4] S. Vellingiri et al., "Experience with a Trans-Pacific Collaborative Mixed Reality Plant Walk," in 2020 IEEE Conference on Virtual Reality and 3D User Interfaces Abstracts and Workshops (VRW), Atlanta, GA, USA: IEEE, Mar. 2020, pp. 238-245.

[5] C. E. Shannon, "A mathematical theory of communication," Bell Syst. Tech. J., vol. 27, no. 3, pp. 379-423, Jul. 1948, doi: 10.1002/j.1538-7305.1948.tb01338.x.

[6] R. B. Rusu and S. Cousins, "3D is here: Point Cloud Library (PCL)," in 2011 IEEE International Conference on Robotics and Automation, Shanghai, China: IEEE, May 2011, pp. 1-4.

[7] J. Duda, "Asymmetric numeral systems: entropy coding combining speed of Huffman coding with compression rate of arithmetic coding." 2013, arXiv: 1311.2540.

[8] P. De Oliveira Rente, C. Brites, J. Ascenso, and F. Pereira, "Graph-Based Static 3D Point Clouds Geometry Coding," IEEE Trans. Multimed., vol. 21, no. 2, pp. 284-299, Feb. 2019, doi: 10.1109/TMM.2018.2859591.

[9] A. F. R. Guarda, N. M. M. Rodrigues, and F. Pereira, "Deep Learning-Based Point Cloud Geometry Coding: RD Control Through Implicit and Explicit Quantization," in 2020 IEEE International Conference on Multimedia & Expo Workshops (ICMEW), London, UK: IEEE, Jul. 2020, pp. 1-6.

[10] Y. Huang et al., "ISCom: Interest-Aware Semantic Communication Scheme for Point Cloud Video Streaming on Metaverse XR Devices," Ieee J Sel Area Comm, vol. 42, no. 4, pp. 1003-1021, Apr. 2024, doi: 10.1109/JSAC.2023.3345430.

[11] S. Wang, J. Jiao, P. Cai, and L. Wang, "R-PCC: A Baseline for Range Image-based Point Cloud Compression," in 2022 International Conference on Robotics and Automation (ICRA), May 2022, pp. 10055-10061.

[12] Fan, Tingyu, et al. "D-dpcc: Deep dynamic point cloud compression via 3d motion prediction." 2022, arXiv: 2205.01135.

[13] E. Alexiou, K. Tung, and T. Ebrahimi, "Towards neural network approaches for point cloud compression," in Applications of Digital Image Processing XLIII, A. G. Tescher and T. Ebrahimi, Eds., Online Only, United States: SPIE, Aug. 2020, p. 4. doi: 10.1117/12.2569115.

[14] G. Fang, Q. Hu, H. Wang, Y. Xu, and Y. Guo, "3DAC: Learning Attribute Compression for Point Clouds," in 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), New Orleans, LA, USA: IEEE, Jun. 2022, pp. 14799-14808. doi: 10.1109/CVPR52688.2022.01440.

[15] A. F. R. Guarda, N. M. M. Rodrigues, and F. Pereira, "Adaptive Deep Learning-Based Point Cloud Geometry Coding," IEEE J. Sel. Top. Signal Process., vol. 15, no. 2, pp. 415-430, Feb. 2021, doi: 10.1109/JSTSP.2020.3047520.

[16] J. Wang, H. Zhu, Z. Ma, T. Chen, H. Liu, and Q. Shen, "Learned Point Cloud Geometry Compression," IEEE Trans. Circuits Syst. Video Technol., vol. 31, no. 12, pp. 4909-4923, Dec. 2021, doi: 10.1109/TCSVT.2021.3051377.

[17] M. Quach, G. Valenzise, and F. Dufaux, "Improved Deep Point Cloud Geometry Compression," in 2020 IEEE 22nd International Workshop on Multimedia Signal Processing (MMSP), Tampere, Finland: IEEE, Sep. 2020, pp. 1-6. doi: 10.1109/MMSP48831.2020.9287077.

[18] R. Gallager, "Low-density parity-check codes," IRE Trans. Inf. Theory, vol. 8, no. 1, pp. 21-28, Jan. 1962, doi: 10.1109/TIT.1962.1057683.

[19] T. M. Cover and J. A. Thomas, Elements of Information Theory. New York, NY, USA: Wiley, 2006.

[20] Sun, Yaping, et al. "Communications, caching, and computing for mobile virtual reality: Modeling and tradeoff." IEEE Transactions on Communications., vol. 67, no. 11, pp. 7573-7586, 2019.

[21] R. Petkova, V. Poulkov, A. Manolova, and K. Tonchev, "Challenges in Implementing Low-Latency Holographic-Type Communication Systems," Sensors, vol. 22, no. 24, p. 9617, Dec. 2022, doi: 10.3390/s22249617.

[22] I. Kozintsev and K. Ramchandran, "Robust image transmission over energy-constrained time-varying channels using multiresolution joint source-channel coding," IEEE Trans. Signal Process., vol. 46, no. 4, pp. 1012-1026, Apr. 1998, doi: 10.1109/78.668553.

[23] D. Gunduz and E. Erkip, "Joint Source–Channel Codes for MIMO Block-Fading Channels," IEEE Trans. Inf. Theory, vol. 54, no. 1, pp. 116-134, Jan. 2008, doi: 10.1109/TIT.2007.911274.

[24] J. Xu, B. Ai, W. Chen, A. Yang, P. Sun, and M. Rodrigues, "Wireless Image Transmission Using Deep Source Channel Coding With Attention Modules," IEEE Trans. Circuits Syst. Video Technol., vol. 32, no. 4, pp. 2315-2328, Apr. 2022, doi: 10.1109/TCSVT.2021.3082521.

[25] H. Huang, L. Cheng, Z. Yu, W. Zhang, Y. Mu, and K. Xu, "Optical Fiber Communication System Based on Intelligent Joint Source-Channel Coded Modulation," J. Light. Technol., pp. 1-9, 2023, doi: 10.1109/JLT.2023.3328311.

[26] E. Bourtsoulatze, D. Burth Kurka, and D. Gündüz, "Deep Joint Source-Channel Coding for Wireless Image Transmission," IEEE Trans. Cogn. Commun. Netw., vol. 5, no. 3, pp. 567-579, Sep. 2019, doi: 10.1109/TCCN.2019.2919300.

[27] L. Sun, Y. Yang, M. Chen, C. Guo, W. Saad, and H. V. Poor, "Adaptive Information Bottleneck Guided Joint Source and Channel Coding for Image Transmission," Ieee J Sel Area Comm, vol. 41, no. 8, pp. 2628-2644, Aug. 2023, doi: 10.1109/JSAC.2023.3288238.

[28] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Las Vegas, NV, USA: IEEE, Jun. 2016, pp. 770-778. doi: 10.1109/CVPR.2016.90.

[29] A. Krizhevsky, I. Sutskever, and G. E. Hinton, "ImageNet Classification with Deep Convolutional Neural Networks," in Advances in Neural Information Processing Systems, Curran Associates, Inc., 2012. Accessed: Aug. 26, 2023. [Online]. Available: https://proceedings.neurips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html

[30] O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation," in Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015, N. Navab, J. Hornegger, W. M. Wells, and A. F. Frangi, Eds., in Lecture Notes in Computer Science. Cham: Springer International Publishing, 2015, pp. 234-241. doi: 10.1007/978-3-319-24574-4_28.

[31] Robert M. Gray and David L. Neuhoff, "Quantization," IEEE Trans. Inf Theory, vol. 44, no. 6, pp. 2325-2383, Oct. 1998, doi: 10.1109/18.720541

[32] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, "Focal Loss for Dense Object Detection".

[33] J. Rissanen and G. Langdon, "Universal modeling and coding," IEEE Trans. Inf. Theory, vol. 27, no. 1, pp. 12-23, Jan. 1981, doi: 10.1109/TIT.1981.1056282.

[34] Ballé, J., Laparra, V., & Simoncelli, "End-to-end optimized image compression", ICLR 2017, Toulon, France.

[35] S. Perry, "JPEG Pleno Point Cloud Coding Common Test Conditions v3.3." ISO/IEC JTC 1/SC 29/WG1 N88044, Jul. 2020.

[36] E. Alexiou, I. Viola, T. M. Borges, T. A. Fonseca, R. L. de Queiroz, and T. Ebrahimi, Eds., "A comprehensive study of the rate-distortion performance in MPEG point cloud compression," APSIPA Trans. Signal Inf. Process., 2019, doi: 10.1017/ATSIP.2019.20.

[37] E. Alexiou, N. Yang, and T. Ebrahimi, "PointXR: A Toolbox for Visualization and Subjective Evaluation of Point Clouds in Virtual Reality," in 2020 Twelfth International Conference on Quality of Multimedia Experience (QoMEX), May 2020, pp. 1-6. doi: 10.1109/QoMEX48832.2020.9123121.

[38] E. Zerman, P. Gao, C. Ozcinar, and A. Smolic, "Subjective and Objective Quality Assessment for Volumetric Video Compression," Electron. Imaging, vol. 31, no. 10, pp. 323-1-323-7, Jan. 2019, doi: 10.2352/ISSN.2470-1173.2019.10.IQSP-323.

[39] D. Tian, H. Ochimizu, C. Feng, R. Cohen, and A. Vetro, "Geometric distortion metrics for point cloud compression," in 2017 IEEE International Conference on Image Processing (ICIP), Beijing: IEEE, Sep. 2017, pp. 3460-3464. doi: 10.1109/ICIP.2017.8296925.

[40] K. Lee, J. Yi, Y. Lee, S. Choi, and Y. M. Kim, "GROOT: a real-time streaming system of high-fidelity volumetric videos," in Proceedings of the 26th Annual International Conference on Mobile Computing and Networking, in MobiCom '20. New York, NY, USA: Association for Computing Machinery, Sep. 2020, pp. 1-14. doi: 10.1145/3372224.3419214

[41] B. Karanov, P. Bayvel, and L. Schmalen, "End-to-End Learning in Optical Fiber Communications: Concept and Transceiver Design," in 2020 European Conference on Optical Communications (ECOC), Brussels, Belgium: IEEE, Dec. 2020, pp. 1-4. doi:

## AUTHOR BIOGRAPHIES

**Wei Zhang** received the B.S. degree in the school of electronic engineering from the Beijing University of Posts and Telecommunications, July, in 2022. He is currently pursuing the Ph.D. degree with the Beijing University of Posts and Telecommunications, Beijing, China. His research interests fall in the optical communication and digital signal processing.

**Zhenming Yu** received the B.Eng. degree in electronic science and technology and the M.Eng. degree in optical engineering from the University of Electronic Science and Technology of China (UESTC), Chengdu, China, in 2011 and 2014, respectively, and the Ph.D. degree in electronic science and technology from Tsinghua University, Beijing, China, in 2018. He was a recipient of the SPIE Optics and Photonics Education Scholarship 2017 and the IEEE Photonics Society Graduate Student Fellowship 2018. He is currently an Associate Professor with the State Key Laboratory of Information Photonics and Optical Communications, Beijing University of Posts and Telecommunications, Beijing, China. His current research interests are optical and digital signal processing for coherent and direct-detection optical communication, intelligent optical communication system, integration of optical perception and communication.

**Hongyu Huang** received the B.S. degree in the school of electronic engineering from the Beijing University of Posts and Telecommunications, July, in 2021. He is currently pursuing the Ph.D. degree with the Beijing University of Posts and Telecommunications. His research interests fall in the optical communication and digital signal processing.

**Xiangyong Dong** is a Ph.D. candidate in the School of Electronic Engineering at Beijing University of Posts and Telecommunications. His research primarily focuses on optical fiber communication and digital signal processing.

**Kaixuan Sun** was born in Fuyang, China. She received the B.E. degree in electronic information engineering from Liaoning Normal University, Dalian, China, in 2019. She is currently pursuing the Ph.D. degree with the State Key Laboratory of Information Photonics and Optical Communications, Beijing University of Posts and Telecommunications, Beijing, China. Her research interests include soft failure management in optical networks.

**Kun Xu** received the B.Sc. degree in applied physics from the Central South University of Technology (currently Central South University), China, in 1996, the M.Sc. degree in optical engineering from the University of Electronic Science and Technology, China, in 1999, and the Ph.D. degree in physical electronics from Tsinghua University, China, in 2004. Then, he joined the Beijing University of Posts and Telecommunications, China. He was a Visiting Scholar with Nanyang Technological University, Singapore, in 2004. He is currently a Professor with the State Key Laboratory of Information Photonics and Optical Communications, Beijing University of Posts and Telecommunications, Beijing, China. He is currently the President of Beijing University of Posts and Telecommunications, Beijing, China. His research interests include fiber-wireless integrated networks, distributed antenna systems, ubiquitous wireless sensing and access, RF photonic integrated systems, artificial intelligence and photonics neural networks, machine learning, and ultrafast optical communication.

Dr. Xu has served for technical program committees (TPC) and workshop/session co-chairs of several international conferences, including the IEEE Microwave Photonics/Asia Pacific Microwave Photonics Conference (MWP/APMP), the IEEE Global Symposium on Millimeter Waves (GSMM), the IEEE International Conference on Communications (ICC), and Progress in Electromagnetics Research Symposium (PIERS). He was also the General Chair of 2020 IEEE/OSA/SPIE Asia Communications and Photonics Conference (ACP 2020), the TPC Co-Chair of ACP 2015 and ACP 2018, and the LOC Chair of ACP 2013 and APMP 2009. He was a Guest Editor of the Special Issue on Microwave Photonics in Photonics Research (OSA/CLP) in 2015.