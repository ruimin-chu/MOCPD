# <em>MOCPD</sup></em>
### Real-time Fuel Leakage Detection via Online Change Point Detection

## Abstract
Early detection of fuel leakage at service stations with underground petroleum storage systems is a crucial task to prevent catastrophic hazards. Current data-driven fuel leakage detection methods employ offline statistical inventory reconciliation, leading to significant detection delays. Consequently, this can result in substantial financial loss and environmental impact on the surrounding community. In this paper, we propose a novel framework called Memory-based Online Change Point Detection (MOCPD) which operates in near real-time, enabling early detection of fuel leakage. MOCPD maintains a collection of representative historical data within a size-constrained memory, along with an adaptively computed threshold. Leaks are detected when the dissimilarity between the latest data and historical memory exceeds the current threshold. An update phase is incorporated in MOCPD to ensure diversity among historical samples in the memory. With this design, MOCPD is more robust and achieves a better recall rate while maintaining a reasonable precision score. We have conducted a variety of experiments comparing MOCPD to commonly used online change point detection (CPD) baselines on real-world fuel variance data with induced leakages, actual fuel leakage data and benchmark CPD datasets. Overall, MOCPD consistently outperforms the baseline methods in terms of detection accuracy, demonstrating its applicability to fuel leakage detection and CPD problems.


## Scripts to Run Each Model

To run different models for the MOCPD (Mean, MMD, and VAE) evaluation, use the following commands:

- For **MOCPD-MEAN**: `python main_mean.py`
- For **MOCPD-MMD**: `python main_mmd.py`
- For **MOCPD-VAE**: `python main_vae.py`
