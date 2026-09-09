# Milestone 2: Conceptual Questions & Answers

**Course**: AI Model Engineering  
**Student / Data Engineer**: Axelle Chandra (24/533796/PA/22614)  
**Topic**: Data Profiling, Quality Auditing, and Preprocessing Methodologies

---

### Question 1: How does hidden missingness (such as biological zero placeholders) compromise machine learning models, and why is domain-specific diagnostic auditing necessary before automated pipeline execution?

**Answer** (138 words):  
Hidden missingness occurs when missing observations are encoded as legitimate numerical values (e.g., zero blood pressure or insulin) rather than explicit null markers. Standard automated preprocessing pipelines fail to detect these because zero is synthetically valid within numeric data types. Consequently, algorithms treat zero as a true physiological state rather than unobserved data. In regression and distance-based classifiers, artificial zeros severely distort summary statistics, shifting sample means downward, inflating sample variance, and generating false negative correlation structures. For instance, computing Euclidean distances with zero insulin assumes hyper-efficient pancreatic function rather than unmeasured physiology. Domain-specific diagnostic auditing prevents this catastrophic distortion by cross-referencing feature definitions against clinical plausibility boundaries, identifying impossible biological states, and mapping them to formal null placeholders before imputation or modeling commences.

---

### Question 2: Why is complete-case analysis (row deletion) suboptimal for high-missingness biomedical tabular data, and what are the trade-offs of class-conditioned median imputation versus multivariate k-NN imputation?

**Answer** (146 words):  
Complete-case analysis (listwise row deletion) discards all records containing at least one missing attribute. In datasets like Pima Indians where insulin exhibits nearly 50% missingness, listwise deletion discards roughly half of all available observations ($N=768 \to 392$), inducing severe sample attrition, loss of statistical power, and potential selection bias if missingness is Missing at Random (MAR) rather than Missing Completely at Random (MCAR). Imputation restores sample completeness. Class-conditioned median imputation preserves group-level distributional separation and is robust against heavy-tailed outliers, but it artificially narrows intra-class variance and cannot easily be applied during inference when true labels are unknown. Conversely, multivariate $k$-NN imputation exploits non-linear covariance across all observed biological features without requiring target labels during test-time screening, but it suffers higher computational complexity ($O(N \cdot D)$) and sensitivity to unscaled feature distances.
