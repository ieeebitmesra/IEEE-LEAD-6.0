<h1>California Housing Median Value Prediction (Normal Equation)</h1>

<p>
This project implements <b>Multiple Linear Regression from scratch using the Normal Equation</b>,
without using scikit-learn.  
I practiced NumPy, Pandas, dataset cleaning, missing value handling, train–test split,
matrix math, and creating visualizations.
</p>

<hr/>

<h2> Project Structure</h2>

<pre>
ML-project1/
│
├── PRACTICE/
│   numpy_prac_1.py
│   prac_pandaas.py
│   linear_regression_prac.py
│   normal_equation_prac.py
│
├── PROJECT
│    California_housing_median.py   ← (Main project script)
│
└── README.md
</pre>

<hr/>

<h2> What this project does</h2>
<ul>
  <li>Loads and explores the <b>California Housing dataset</b></li>
  <li>Handles missing values using <b>median imputation</b></li>
  <li>Performs <b>80:20 random train–test split</b> (without sklearn)</li>
  <li>Implements <b>Multiple Linear Regression using the Normal Equation</b></li>
  <li>Calculates <b>Train MSE</b>, <b>Test MSE</b>, and <b>R² Score</b></li>
  <li>Visualizes <b>Median Income vs Median House Value</b></li>
</ul>

<hr/>

<h2> Results</h2>
<ul>
  <li><b>Train MSE:</b> 0.52359</li>
  <li><b>Test MSE:</b> 0.529377</li>
  <li><b>Test R² Score:</b> 0.614</li>
</ul>

<p>
These values show that the model explains <b>~61% of the variance</b> in house prices
using the selected features.
</p>

<hr/>

<h2> Plot Interpretation</h2>

<h3>Median Income vs Median House Value</h3>
<p>
The scatter plot shows that:
</p>
<ul>
  <li>As <b>median income increases</b>, <b>median house value increases</b> too.</li>
  <li>The relationship is <b>positively correlated</b> but not perfectly linear.</li>
  
</ul>

<p>
This confirms that <b>income is one of the strongest predictors</b> of house price.
</p>

<hr/>

<h2> How to Run</h2>

<pre>
python -m venv .venv
source .venv/bin/activate       # On Windows → .venv\Scripts\activate
pip install numpy pandas matplotlib
python PROJECT/California_housing_median.py
</pre>

<hr/>

<h2> Learnings</h2>
<ul>
  <li>Practiced NumPy operations & Pandas basics</li>
  <li>Data cleaning & handling missing values</li>
  <li>Implemented <b>normal equation</b> manually</li>
  <li>Built multiple linear regression without sklearn</li>
  <li>Created visualization using matplotlib</li>
  <li>Understood matrix multiplication & evaluation metrics</li>
</ul>

<hr/>

<h2> Notes</h2>
<ul>
  <li>No ML libraries like sklearn were used</li>
  <li>Only <b>NumPy + Pandas + Matplotlib</b></li>
  <li>This is my <b>first end-to-end ML project</b></li>
</ul>



