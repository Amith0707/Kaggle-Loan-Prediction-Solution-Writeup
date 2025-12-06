#  Loan Default Prediction — My First Kaggle Competition Journey 
**Competition Link**-https://www.kaggle.com/competitions/playground-series-s5e11/overview 

**Rank:** *751 / 3724*  
**Best Score:** *0.92497 AUC (CatBoost)*  
**Competition:** *Kaggle Playground Series – Season 5 Episode 11*

---

## 1️⃣ Introduction — What Started as a Model Experiment Became a Data-Science Reality Check  
This was my **first-ever Kaggle competition**, and I initially approached it like a typical ML assignment:

> Just choose a model, tune hyperparameters, and submit.

But the competition quickly taught me something deeper:

### 1. **Modeling is not the main differentiator**  
### 2. **Feature engineering mattered more than the model itself**  
### 3. **Small decisions in preprocessing can move your score by 0.01+ AUC**

This project became the turning point where I truly understood what it means to *engineer* a solution, not just train a model.

---

## 2️⃣ Key Things I Learned (The Reality Behind a Good Kaggle Score)

### **1. Feature Engineering Matters More Than I Expected**  
Initially I did simple one-hot encoding and used LightGBM.  
Result: **~0.922 AUC**

But when I started experimenting with:  
- Frequency Encoding  
- Count Encoding  
- Ratio Features  
- Debt/Income relationships  
- Smoothing-based Target Encoding  

My score jumped significantly.

I learned that **your model can only be as good as the features you feed it**.

---

### **2. Cross-Validation Isn’t Optional — It’s Survival**  
In the beginning, I trained on full data and prayed the leaderboard liked it.  
Then I learned:

- How to properly use **Stratified KFold**  
- Why **OOF (Out-of-Fold)** predictions matter  
- How leakage can destroy leaderboard score  
- How CV ensures stability across folds  

This gave me a stable, reliable score instead of jumping up and down the leaderboard.

---

### **3. Optuna & Hyperparameter Tuning (Beyond GridSearchCV)**  
Before this competition, I only knew:

- GridSearchCV  
- RandomizedSearchCV  

Then I discovered **Optuna**, a far more powerful hyperparameter optimizer.

I learned:  
- How Optuna searches intelligently  
- Why Bayesian-like optimization works better  
- How to integrate Optuna with LightGBM/XGBoost  

This gave me the confidence to tune models at a much deeper level.

---

### **4. Stacking & Blending Models**  
I experimented with combining:  
- LightGBM  
- XGBoost  
- CatBoost  

Simple blending gave me small gains.  
But stacking made me understand how:

- Different models capture different patterns  
- Ensembles create robustness  
- Weighted blends can outperform individual models  

---

### **5. The Final Breakthrough Came from CatBoost + Frequency Encoding**  
After many experiments, the biggest improvement came from:

- Cleaning categorical features  
- Applying **frequency count encoding**  
- Feeding it to **CatBoost** with tuned hyperparameters  

This finally pushed me to **0.92497 AUC**, my highest score.

---

## 3️⃣ My Initial Approach — The Baseline That Started Everything

Before reaching the 0.92497 CatBoost solution, I went through a full cycle of experimentation.  
This section summarizes **what I tried first**, what worked, and what didn't — and how those attempts shaped my final strategy.

---

### **1. LightGBM Baseline (First Attempt)**  
Like most Kaggle beginners, I started with a standard pipeline:

- One-hot encode categorical features  
- Train a LightGBM model with default parameters  
- Use 5-fold Stratified CV  

**Score:** ~0.9221 AUC

This was decent but not competitive.  
I noticed the model wasn't capturing deeper interactions or categorical patterns.

This led me to try improving the model rather than the data.

---

###  **2. Hyperparameter Tuning with Optuna (XGBoost + LightGBM)**  
Next, I explored **Optuna**, which was new to me.

I ran separate tuning sessions for:

- **XGBoost** (tree_method = hist, GPU)  
- **LightGBM** (GPU, large estimators)

I learned:

- Optuna finds much better hyperparameters than GridSearchCV  
- XGBoost responded strongly to depth, gamma, and child_weight  
- LightGBM improved with tuned num_leaves and learning rate  

After tuning both, I got:

- **XGBoost OOF:** ~0.9224  
- **LightGBM OOF:** ~0.9227  

These were stable but still not enough to break 0.923+.

---

### **3. Blending / Stacking These Models**  
My intuition was:

> “If XGBoost and LightGBM capture different relationships, combining them should provide a gain.”

So I created:

- A weighted blend (optimized weight: ~0.14 for XGB)  
- A simple stacked model with meta-learner  

But the improvement was extremely small:

- **Blended OOF:** ~0.9229  
- **Public LB:** ~0.9231 (my previous best)

This was the moment I realized:

###  *The models were already strong — the features were not.*

This pushed me into exploring feature engineering and non-linear encodings.

---

## 4️⃣ How These Attempts Led to My Final Approach

After the experiments above, I had a big realization:

### **If multiple strong models all plateau at ~0.922–0.923,the limitation is in the data representation, not the model.**

This intuition came from noticing:

- One-hot encoding was too sparse  
- Many categorical features had meaningful frequency patterns  
- Ratios like loan_amount / annual_income were not captured  
- Debt-related interactions were missing  
- Raw features lacked signal transformation  
- CatBoost handles categorical structures more naturally

So I shifted my focus from:

**Tune the model harder**  
to  
**Engineer the data smarter**

This mindset change led directly to:

- Frequency / count encoding  
- Ratio features  
- Debt/income engineering  
- Simpler but more expressive feature representation  
- CatBoost (best suited for categorical-heavy datasets)

And that became the foundation for the final solution that scored **0.92497 AUC**.

---

## 5️⃣ Final Approach (Step-by-Step Summary)

###  Step 1: Preprocessing
- Identified categorical and numerical features  
- Applied **frequency encoding** to categorical columns  
- Removed original categorical columns  
- Used training distribution for all mappings  

###  Step 2: Feature Engineering
- Created debt-related ratios  
- Added intuitive financial heuristics  
- Ensured no leakage  

###  Step 3: Model Training (CatBoost)
```python
model = CatBoostClassifier(
    iterations=30000,
    learning_rate=0.024,
    depth=3,
    l2_leaf_reg=5,
    random_strength=2.5,
    bagging_temperature=5.0,
    border_count=450,
    grow_policy='Depthwise',
    boosting_type='Plain',
    eval_metric='AUC',
    early_stopping_rounds=500,
    eval_fraction=0.2,
    verbose=500,
    random_seed=42,
    use_best_model=True,
    od_type='Iter'
)
model.fit(X, y)
```

### Inference & Submission

```python
ts_proba = model.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({
    "id": test_ids,
    "loan_paid_back": ts_proba
})

submission.to_csv("submission.csv", index=False)
```
