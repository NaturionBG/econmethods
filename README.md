# <b>econmethods</b>

# A library made specifically for an econometrics project. 
<h1>Contains:</h1>
<ul>
  <li>The one-way CIPS test, accesible via the CipsTest class.</li>
  <li>The two-way Hausman test, accesible via the Hausman class</li>
  <li>A universal class for estimating a first-order Error Correction Model, accesible via the FECM class. Hyperparameters include:</li>
  <ol>
    <li> turn on/off automatic recursive removal of insignificant coefficients. The significance level is set by the user.</li>
    <li> turn on/off the addition of constants for both the long-run model and the short-run model.</li>
    <li> choose the kind of effects for the long run model (RE/FE), choose the ECM estimation strategy (MG, CCEP, CCEMG)</li>
    <li> augment with a trend (if exists in the user's data), choose the variables that make up the cointegrating vector, or add static variables into the ECM estimation, if economic theory suggests that such insertion is plausible. Such variables will be inserted into the estimation in their original levels (or log-levels), not in their first differences. </li>
  </ol>
  <li>The Cross-Sectional Dependence test, accesible via the CDTest class.</li>
  <li>The Slope Homogeneity F-test, accesible via the SlopeHomogeneityF class.</li>
</ul>
