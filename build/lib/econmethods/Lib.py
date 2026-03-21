import numpy as np
import pandas as pd
import statsmodels.api as sm
from itertools import product, combinations
import scipy.stats as sc
from math import floor
import importlib.resources as resources
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.stats.diagnostic import het_goldfeldquandt
from typing import Any

def read_critical_values(sheet: str) -> pd.DataFrame:
    xlsx_path = resources.files("econmethods") / "CADF_Crit_Values.xlsx"
    df = pd.read_excel(xlsx_path, sheet_name=sheet, index_col=0)
    return df

class CipsTest:
  '''
  Implementation of the standard Cross-Sectionally Augmented Dickey-Fuller 
  procedure to test for non-stationarity I(1) in panel data. Works only with a linear trend.
  ----
  The hypotheses are as follows:
  - *H0*: The Target variable is I(1)
  - *H1*: The Target variable is I(0)\n
  PARAMETERS:
  ----------
  ------
  - *df*: A standart Pandas DataFrame containing panelized data. \n
    Ensure that the DataFrame contains the following columns in this exact order: \n
    0 - a column of spatial units. Must contain homogenous data, e.g. only countries, companies, regions, etc. \n
    1 - temporal column. Must contain homogenous data, e.g. only years, months, quarters, etc. \n
    2 - target variable. Must not contain NaN Values. An Error will be raised otherwise. \n
  
  - *T*: Your Temporal window. Will be used to determine the test critical value.
  - *N*: Your Spatial window. Will be used to determine the test critical value.
  
  - *trend*: State whether your ADF model has a trend or not. Will be used to determine the test critical value.
  
  - *poly_trend*: State whether your target variable has a ditinct polynomial trend. \n
    If a value bigger than 1 is entered, the test will detrend the target variable to get a robust result.
  
  - *intercept*: State whether ADF model has an intercept or not. Will be used to determine the test critical value.
  
  - *n_lags*: Determine the maximum amount of lags in the Augmented Dickey-Fuller regression.
    the test will choose the best lag amount from 1 to n_lags based on AIC (Akaike Information Criterion) 
    
  - *level*: Value of significance to conduct the test at (in %%). Only 5 and 1% are allowed.
  - *log_target*: Specify whether the test shall aplly a natural logarithm to your target variable. Defaults to True.
  -----------------------------
  RETURNS:
  --
  via instance.verdict() -> None: Prints the verdict of the test based on the parameters set by the user.
  '''
  def __init__(self, df: pd.DataFrame, T: int, N: int, trend: bool =  False, poly_trend: int = 1, intercept: bool = False, n_lags: int = 2, level: int = 5, log_target: bool = True) -> None:
    CipsTest.__build_tables()
    self.__df = df
    self.__L = 1
    self.__T = T
    self.__N = N
    self.__trend = trend
    self.__C = intercept
    self.__n_lags = n_lags
    if self.__n_lags > floor(self.__T/5):
      self.__n_lags = floor(self.__T/5)
      if self.__n_lags < 1:
        self.__n_lags = 1
    self.__alpha = level/100
    self.__df = self.__df.rename(columns={self.__df.columns[0]:'SpUnit', self.__df.columns[1]:'time', self.__df.columns[2]:'target'})
    if log_target:
      self.__df.target = np.log(self.__df.target)
    self.__poly = poly_trend
    if self.__trend:
      if self.__poly > 1:
        self.__df = self.detrend()
        self.__trend = False
    self.verify()
    self.__table = self.get_table()
    self.__CADF_Crit = self.get_critical_value()
    self.__CADF = self.estimate()
  
  def verify(self) -> None:
    if self.__df.target.isnull().sum() > 0:
      raise TypeError('Values in Target must NOT be NaN!')
    if self.__alpha != 0.01 and self.__alpha != 0.05:
      raise ValueError('The Significance Level must be either 1 or 5!')
    if self.__poly < 1:
      raise ValueError('The Polynomial Power Cannot be lesser than 1!')
      
  @classmethod
  def __build_tables(cls) -> None:
    cls.NTNC_1P = read_critical_values('NTNC_1P')
    cls.NTNC_5P = read_critical_values('NTNC_5P')
    cls.NTC_1P = read_critical_values('NTC_1P')
    cls.NTC_5P = read_critical_values('NTC_5P')
    cls.TC_1P = read_critical_values('TC_1P')
    cls.TC_5P = read_critical_values('TC_5P')
    
  def get_table(self) -> pd.DataFrame:
    if not self.__trend and not self.__C:
      if self.__alpha == 0.01:
        return CipsTest.NTNC_1P
      else:
        return CipsTest.NTNC_5P
    if not self.__trend and self.__C:
      if self.__alpha == 0.01:
        return CipsTest.NTC_1P
      else:
        return CipsTest.NTC_5P
    if self.__trend and self.__C:
      if self.__alpha == 0.01:
        return CipsTest.TC_1P
      else:
        return CipsTest.TC_5P
  
  def get_critical_value(self) -> float:
    dct = {}
    for arr in product(self.__table.index, self.__table.index):
      lst = np.array(arr)
      dt = np.array([self.__T, self.__N])
      dct[arr] = np.sqrt(np.sum((dt-lst)**2))
    return self.__table.loc[min(dct, key=dct.get)]

  def detrend(self) -> pd.DataFrame:
    lst = []
    for unit in self.__df.SpUnit.unique():
      subdf = self.__df[self.__df.SpUnit == unit]
      for i in range(1, self.__poly+1):
        subdf.insert(3, f't^{i}',  np.linspace(1, len(subdf), len(subdf))**i)
      diff = subdf['target'].copy() - sm.OLS(subdf['target'], sm.add_constant(subdf.iloc[:, 3:])).fit().predict(sm.add_constant(subdf.iloc[:, 3:]))
      subdf.loc[:, 'target'] = diff
      lst.append(subdf.iloc[:, :3])
    return pd.concat(lst, axis=0)
        
  def build_regressions(self, lags: int) -> list[pd.DataFrame]:
    lst = []
    for unit in self.__df.SpUnit.unique():
      subdf = self.__df[self.__df.SpUnit == unit]
      if self.__trend:
        subdf.insert(2, 't', np.linspace(1, len(subdf), len(subdf)))
      subdf = pd.concat([subdf, subdf.target.shift(periods=range(1, lags+1))], axis=1)
      subdf['cs_avg'] = self.__df.groupby(['time'])['target'].mean().values
      subdf = pd.concat([subdf, subdf.cs_avg.shift(periods=range(1, lags+1))], axis=1)
      subdf.insert(3, 'target_diff', subdf.target - subdf.target_1)
      subdf['cs_avg_diff'] = subdf.cs_avg - subdf.cs_avg_1
      subdf = pd.concat([subdf, subdf.cs_avg_diff.shift(periods=range(1, lags+1))], axis=1)
      subdf = pd.concat([subdf, subdf.target_diff.shift(periods=range(1, lags+1))], axis=1)
      if self.__trend:
        base = ['target_diff', 't', 'target_1', 'cs_avg_1', 'cs_avg_diff']
      else:
        base = ['target_diff', 'target_1', 'cs_avg_1', 'cs_avg_diff']
      additional = []
      for i in range(1, lags+1):
        additional.append(f'target_diff_{i}')
        additional.append(f'cs_avg_diff_{i}')
      subdf=(subdf.loc[:, base+additional]).iloc[self.__n_lags+1:, :]
      lst.append(subdf)
    return lst

  def estimate(self) -> float:
    CADF_stat = None
    best_aic = np.inf
    for l in range(1, self.__n_lags+1):
      aic = []
      lst1 = self.build_regressions(l)
      CADF = []
      for frame in lst1:
        if self.__C:
          res = sm.OLS(frame.iloc[:, 0], sm.add_constant(frame.iloc[:, 1:])).fit()
        else:
          res = sm.OLS(frame.iloc[:, 0], frame.iloc[:, 1:]).fit()
        CADF.append(res.tvalues['target_1'])
        aic.append(res.aic)
      if len(np.unique(np.array(CADF))) != 1:
        CADF = np.array(CADF).mean()
        if np.array(aic).mean() < best_aic:
          CADF_stat = CADF
          best_aic = np.array(aic).mean()
          self.__L = l
    return CADF_stat
  
  def verdict(self) -> None:
    if self.__CADF < self.__CADF_Crit:
      print(f'{self.__CADF} < {self.__CADF_Crit}\n Your target variable is I(0) according to the CIPS test\n Significance level : {self.__alpha*100}%. \n Selected lag amount: {self.__L}')
    else:
      print(f'{self.__CADF} > {self.__CADF_Crit}\n Your target variable is I(1) according to the CIPS test\n significance level: {self.__alpha*100}%. \n Selected lag amount: {self.__L}')
  
  def __del__(self) -> None:
    pass


class HausmanOneWay:
  '''
  Implementation of the Hausman procedure to test whether it is more feasible to used random effects against fixed effects.
  -
  The Hypotheses are as follows
  - *H0*: Cov(a_i, x_{it}) = 0
  - *H1*: Cov(a_i, x_{it}) != 0\n
  PARAMETERS:
  --
  - *data*: A pandas DataFrame. Make sure that all items are introduced in this exact order by column index:\n
    Make sure no columns contain Nan Values!
    0 - your spatial unit. The data must be homogenous, e.g. only contries, regions, etc.
    1 - your temporal window per panel. The data must be homogenous, e.g. only years, months, etc.
    2 - Your target variable.
    3+ - your exogenous variables.
  - *level*: the statistical test significance level.
  - *force_GLS*: Choose if MLE diverges.
  ---------
  METHODOLOGY:
  ------
  - The Test prefers an MLE estimator for RE (Random Effects), 
    which can be disrupted if the difference between FE and RE is considerably small. 
    The test will opt for MLE estimation if sigma_u computed for GLS is greater than 0.
------
  RETURNS:
  -
  - Via instance.verdict() prints the verdict of the Hausman test according to the given parameters.
  '''
  def __init__(self, data: pd.DataFrame, force_GLS: bool = False, level: int = 5) -> None:
    self.__df = data
    self.__exog = len(data.columns[3:])
    self.__l =[]
    self.__fgls = force_GLS
    for i in range(1, self.__exog+1):
      self.__l.append(f'x{i}')
    self.__df.columns = ['SpUnit', 'time', 'target'] + self.__l
    self.__alpha = level/100
    self.__RE = 'GLS'
    self.__T = len(self.__df.time.unique())
    self.__N = len(self.__df.SpUnit.unique())
    
    
  def build_FE(self) -> pd.DataFrame:
    fe = self.__df.copy(deep=True)
    for i, unit in enumerate(self.__df.SpUnit.unique()[1:]):
      fe.loc[:, f'd{i}'] = np.where(fe.SpUnit == unit, 1, 0)
    return fe.iloc[:, 2:]
  
  def build_GLS(self, w_err: float) -> pd.DataFrame | None:
    re = self.__df.copy(deep=True)
    sigma2 = np.sum((sm.OLS(re.iloc[:, 2], sm.add_constant(re.iloc[:, 3:])).fit()).resid**2) / (self.__N*self.__T - self.__exog)
    sigma_u = sigma2 - w_err
    if sigma_u <= 0:
      sigma_u = 0
    elif sigma_u > 0:
      if not self.__fgls:
        self.__RE = 'MLE'
        return None
    sig = np.full((self.__T, self.__T), sigma_u)
    np.fill_diagonal(sig, sigma2)
    matrix = np.kron(np.eye(self.__N), sig)
    return matrix
      
      
  def estimate(self) -> float:
    fe_df = self.build_FE()
    Chi = 0
    fe_res = sm.OLS(fe_df.iloc[:, 0], sm.add_constant(fe_df.iloc[:, 1:])).fit()
    matrix = self.build_GLS(np.sum(fe_res.resid**2) / (self.__N*(self.__T-1) - self.__exog))
    if matrix is not None:
      re_res = sm.GLS(self.__df.iloc[:, 2], sm.add_constant(self.__df.iloc[:, 3:]), matrix).fit()
    else:
      re_res = sm.MixedLM(self.__df['target'], sm.add_constant(self.__df[self.__l]), groups=self.__df.SpUnit).fit(reml = True, maxiter=100_00)
    for b_fe, b_re, var_fe, var_re in zip(fe_res.params[1:], re_res.params[1:], fe_res.bse[1:]**2, re_res.bse[1:]**2):
      Chi += (b_fe - b_re)**2 / (var_fe - var_re)
    
    return Chi
  
  def verdict(self) -> None:
    Chi = self.estimate()
    ch2 = sc.chi2(self.__exog)
    p = ch2.sf(Chi)
    if p < self.__alpha:
      print(f'P-Value: {p} < Alpha: {self.__alpha}. \n According to the Hausman test, you should use the FE (Fixed Effects) model. \n RE estimator: {self.__RE}')
    else:
      print(f'P-Value: {p} > Alpha: {self.__alpha}. \n According to the Hausman test, you should use the RE (Random Effects) model. \n RE estimator: {self.__RE}')
  
  def __del__(self) -> None:
    pass


class FECM:
  '''
  The implementation of a first-order ECM (Error Correction Model) estimation for panel data.
  ----
  ----
  PARAMETERS:
  ----
  - *df*: a Pandas DataFrame containing panel data. Make sure your data is structured in this exact order (by column index):\n
    0 - Spatial units. The data must be homogenous, e.g. only cities, contries, regions, etc.\n
    1 - Temporal units. The data must be homogenous, e.g. only years, months, quarters, etc. \n
    2 - Your target/endogenous variable. The data must not contain NaN values.\n
    3+ - Your exogenous variables. The data must not contain NaN values.\n
  - *effects*: Specify what effects your long-run model must have. The class will estimate both the short-run & long-run models.
    Enter one of the following keywords: "fix" | "rand". "rand" by default.
  - *trend*: Specify a trend of which order your target variable is. 0 by default - the data contains no trend.
  - *n_lags*: Specify the maximum amount of lags to test on.
  - *method*: Specify the method of ECM estimation. \n
    Currently implemented methods:\n
    - MG (Mean Group) - Choose this method of you believe all your coefficients may be simply heterogenous.
    - CCEMG (Common Correlated Effects Mean Group) - Choose this one if you also believe that there is valid cross-sectional dependence in the data.
    - CCEP (Common Correlated Effects Pooled) - Choose this one either if your data is lacking temporally or if you believe in the homogeneity of regressors.\n
    Choose between the following keywords: ["MG", "CCEMG", "CCEP"]
    - If CCE- is chosen, the mean target variable for forecasting the differences will be predicted using an AR(d) proccess, where d will be chosen automatically between 1 and n_lags.
  - *coint*: Specify which exogenous variables will be included in the long-run model (A.K.A. are conitegrated with the TARGET variable). 
      The rest will be included only in the short-run model.\n
      Enter: a string containing 1 single variable / a list of strings in the following format.\n
      (enumeration from left to right column-wise in your DataFrame):
      - coint = "x1"
      - coint = ["x1", "x3", ...]\n
      Defaults to "x1"
  - *include_x_diffs*: Specify whether the model should include the differences of exogenous variables. Defaults to True.
  - *intercept*: Specify whether the ECM model should have an intercept. Defaults to True.
  - *stat_vars*: a DataFrame of the same format as "df" - includes variables that will not be differenced and included into the ECM in their raw form. Ensure these variables are I(0). Defaults to None.
  - *lr_const*: Specify whether the long-run model should have a constant.
  ----
  RETURNS:
  --
  A python dictionary (dict) containing:
    - Long-run estimation results | key = "lr_res"
    - ECM (short-run) estimation results | key = "sr_res"\n
    If a CCE- method is chosen:
    - The AR(d) estimation results to forecast the cross-sectional mean | key = "ar"
  '''
  def __init__(self, df: pd.DataFrame, effects: str = 'rand', trend: int = 0, n_lags: int = 1, method: str = 'MG', coint: str | list[str] = 'x1', include_x_diffs: bool = True, intercept: bool = True, stat_vars: pd.DataFrame|None = None, lr_const: bool = False) -> None:
    self.__df = df
    self.__eff = effects.lower()
    self.__t = trend
    self.__C = intercept
    self.__lag = n_lags
    self.__method = method.lower()
    self.__exog = len(df.columns[3:])
    self.__l =[]
    self.__stat_vars = stat_vars
    self.__mean_names = ['target_avg']
    self.__x_difs = include_x_diffs
    self.__stat = []
    self.__lr_c = lr_const
    if stat_vars is not None:
      for i in range(1, len(stat_vars.columns[2:])+1):
        self.__stat.append(f'stat{i}')
      self.__stat_vars.columns = ['SpUnit', 'time'] + self.__stat
    for i in range(1, self.__exog+1):
      self.__l.append(f'x{i}')
      self.__mean_names.append(f'x{i}_avg')
    self.__df.columns = ['SpUnit', 'time', 'target'] + self.__l
    if isinstance(coint, list):
      self.__lr_df = self.__df.copy(deep=True).loc[:, ['SpUnit', 'time', 'target', *coint]]
    elif isinstance(coint, str):
      self.__lr_df = self.__df.copy(deep=True).loc[:, ['SpUnit', 'time', 'target', coint]]
    else:
      raise TypeError('An invalid Type has been passed into the COINT parameter')
    if self.__t > 0:
      self.__lr_df = self.add_trend()
    self.__N = len(self.__df.SpUnit.unique())
    self.__T = len(self.__df.time.unique())
    if self.__lag > self.__T**(1/3):
      self.__lag = floor(self.__T**(1/3))
      if self.__lag < 1:
        self.__lag = 1
    self.__verify()
    self.__means = self.build_means()
    if self.__method == 'ccemg' or method == 'ccep':
      self.__ar = self.select_ar()
    self.__lr = self.__estimate_lr()
    self.__sr = self.build_sr()
  
  def __verify(self) -> None:
    if self.__eff not in ['fix', 'rand']:
      raise ValueError('Non-Valid panel effects type entered!')
    if self.__t < 0:
      raise ValueError('The Trend order cannot be lower than 0!')
    if self.__method not in ['mg', 'ccemg', 'ccep']:
      raise NotImplementedError('Either the estimation method has not been implemented yet, or it is invalid!')
    
  def add_trend(self) -> pd.DataFrame:
    lst = []
    for unit in self.__df.SpUnit.unique():
      subdf = self.__lr_df[self.__lr_df.SpUnit == unit].copy(deep=True)
      for i in range(1, self.__t+1):
          subdf.loc[:, f't^{i}'] = np.linspace(1, len(self.__df.time.unique()), len(self.__df.time.unique()))**i
      lst.append(subdf)
    return pd.concat(lst)
        
  def build_means(self) -> pd.DataFrame:
    mn = self.__df.copy(deep=True)
    mn = mn.set_index('time')
    means = mn.groupby('time')[['target'] + self.__l].mean()
    means.columns = self.__mean_names
    means = pd.concat([means, means.shift([1])], axis=1)
    for var in means.columns:
      if 'target' not in var:
        if self.__x_difs:
          means[f'{var}_diff'] = means[var] - means[f'{var}_1']
          means = means.drop(columns=[f'{var}_1'])
      else:
        means[f'{var}_diff'] = means[var] - means[f'{var}_1']
        means = means.drop(columns=[f'{var}_1'])
    return means
  
  def build_GLS(self, w_err: float) -> np.ndarray:
    re = self.__lr_df.copy(deep=True)
    sigma2 = np.sum((sm.OLS(re.iloc[:, 2], sm.add_constant(re.iloc[:, 3:])).fit()).resid**2) / (self.__N*self.__T - self.__exog)
    sigma_u = sigma2 - w_err
    if sigma_u <= 0:
      sigma_u = 0
    sig = np.full((self.__T, self.__T), sigma_u)
    np.fill_diagonal(sig, sigma2)
    matrix = np.kron(np.eye(self.__N), sig)
    return matrix
  
  def build_FE(self) -> pd.DataFrame:
    lr = self.__lr_df.copy(deep=True)
    for i, unit in enumerate(lr.SpUnit.unique()[1:], start=1):
        lr[f'd{i}'] = np.where(lr.SpUnit == unit, 1, 0)
    return lr
  
  def __estimate_lr(self) -> pd.DataFrame:
    if self.__eff == 'fix':
      lr_fe = self.build_FE()
      if self.__lr_c:
        res_lr = sm.OLS(lr_fe.loc[:, 'target'], sm.add_constant(lr_fe.iloc[:, 3:])).fit()
      else:
        res_lr = sm.OLS(lr_fe.loc[:, 'target'], lr_fe.iloc[:, 3:]).fit()
      return res_lr
    else:
      lr_fe = self.build_FE()
      if self.__lr_c:
        resid = np.sum(sm.OLS(lr_fe.loc[:, 'target'], sm.add_constant(lr_fe.iloc[:, 3:])).fit().resid**2) / (self.__N*(self.__T - 1) - self.__exog)
        lr_re_matrix = self.build_GLS(resid)
        return sm.GLS(self.__lr_df.loc[:, 'target'], sm.add_constant(self.__lr_df.iloc[:, 3:]), lr_re_matrix).fit()
      else:
        resid = np.sum(sm.OLS(lr_fe.loc[:, 'target'], lr_fe.iloc[:, 3:]).fit().resid**2) / (self.__N*(self.__T - 1) - self.__exog)
        lr_re_matrix = self.build_GLS(resid)
        return sm.GLS(self.__lr_df.loc[:, 'target'], self.__lr_df.iloc[:, 3:], lr_re_matrix).fit()
  
  def select_ar(self) -> Any:
    current_d = self.__lag+1
    while current_d >= 1:
      frame = pd.DataFrame(self.__means.target_avg)
      temp = []
      for lag in range(1, current_d+1):
        frame.loc[:, f'y_avg{lag}'] = frame['target_avg'].shift(lag)
        temp.append(f'y_avg{lag}')
      frame = frame.dropna()
      part_res = sm.OLS(frame.target_avg, frame[temp]).fit()
      if part_res.pvalues[temp[-1]] < 0.05:
        break
      else:
        current_d -=1
    print(f'Selected AR lag amount: {current_d}')
    return part_res
  
  def get_ccemg_frames(self, max_lag: int) -> list[pd.DataFrame]:
    subdfs = []
    for unit in self.__df.SpUnit.unique():
      subdf = self.__df[self.__df.SpUnit == unit].copy(deep=True)
      if self.__x_difs:
        for var in self.__l:
          subdf[f'{var}_lag1'] = subdf[var].shift(1)
          subdf[f'{var}_diff'] = subdf[var]- subdf[f'{var}_lag1']
          subdf = subdf.drop(columns = [f'{var}_lag1'])
      subdf['target_lag1'] = subdf['target'].shift(1)
      subdf.insert(2, 'target_diff', subdf['target'] - subdf['target_lag1'])
      subdf = subdf.drop(columns = ['target_lag1', *self.__l, 'target'])
      subdf['error'] = subdf.error.shift(1)
      subdf = pd.concat([subdf.reset_index(drop=True), self.__means.reset_index(drop=True)], axis=1)
      if self.__stat_vars is not None:
        stat_subdf = self.__stat_vars[self.__stat_vars.SpUnit == unit].copy(deep=True)
        subdf = pd.concat([subdf.reset_index(drop=True), stat_subdf.iloc[:, 2:].reset_index(drop=True)], axis=1)
      subdfs.append(subdf.dropna())
    return subdfs
      
  def get_mg_frames(self, max_lag: int) -> list[pd.DataFrame]:
    subdfs = []
    for unit in self.__df.SpUnit.unique():
      subdf = self.__df[self.__df.SpUnit == unit].copy(deep=True)
      if self.__x_difs:
        for var in self.__l:
          subdf[f'{var}_lag1'] = subdf[var].shift(1)
          subdf[f'{var}_diff'] = subdf[var]- subdf[f'{var}_lag1']
          subdf = subdf.drop(columns = [f'{var}_lag1'])
      subdf['target_lag1'] = subdf['target'].shift(1)
      subdf.insert(2, 'target_diff', subdf['target'] - subdf['target_lag1'])
      subdf = subdf.drop(columns = ['target_lag1', *self.__l, 'target'])
      subdf['error'] = subdf.error.shift(1)
      if self.__stat_vars is not None:
        stat_subdf = self.__stat_vars[self.__stat_vars.SpUnit == unit].copy(deep=True)
        subdf = pd.concat([subdf.reset_index(drop=True), stat_subdf.iloc[:, 2:].reset_index(drop=True)], axis=1)
      subdfs.append(subdf.dropna())
    return subdfs
      
  def build_sr(self) -> pd.DataFrame:
    self.__df = pd.concat([self.__df, pd.Series(self.__lr.resid, name='error')], axis=1)
    est = []
    if self.__method == 'ccemg':
      units = self.get_ccemg_frames(self.__lag)
      for model in units:
        if self.__C:
          est.append(sm.OLS(model['target_diff'], sm.add_constant(model.iloc[:, 3:])).fit())
        else:
          est.append(sm.OLS(model['target_diff'], model.iloc[:, 3:]).fit())
      return est
    elif self.__method == 'mg':
      units = self.get_mg_frames(self.__lag)
      for model in units:
        if self.__C:
          est.append(sm.OLS(model['target_diff'], sm.add_constant(model.iloc[:, 3:])).fit())
        else:
          est.append(sm.OLS(model['target_diff'], model.iloc[:, 3:]).fit())
      return est
    elif self.__method == 'ccep':
      units = self.get_ccemg_frames(self.__lag)
      pool = pd.concat(units, axis=0)
      if self.__C:
        est.append(sm.OLS(pool['target_diff'], sm.add_constant(pool.iloc[:, 3:])).fit())
      else:
        est.append(sm.OLS(pool['target_diff'], pool.iloc[:, 3:]).fit())
      return est

  def fit(self) -> dict:
    dct = dict()
    if self.__method == 'ccep':
      dct['sr_res'] = self.__sr[0]
      dct['lr_res'] = self.__lr
      dct['ar'] = self.__ar
    elif self.__method == 'ccemg' or self.__method == 'mg':
      dct['lr_res'] = self.__lr
      if self.__method == 'ccemg':
        dct['ar'] = self.__ar
      coefs = []
      rsq = []
      for result in self.__sr:
        coefs.append(result.params)
        rsq.append(result.rsquared)
      coef_mean = pd.concat(coefs, axis=1).mean(axis=1)
      coef_mean.name = 'Mean Coefs'
      coef_std = pd.concat(coefs, axis=1).std(axis=1)
      coef_mse = coef_std/np.sqrt(self.__N)
      t_means = coef_mean / coef_mse
      mg_W = sc.chi2(self.__exog).sf(np.sum(coef_mean**2 / coef_mse**2))
      tpvalues_mean = t_means.apply(lambda x: 2*min(sc.t(self.__N-1).cdf(x), sc.t(self.__N-1).sf(x)))
      tpvalues_mean.name = 'T-pvalues'
      rsq_mean = np.array(rsq).mean()
      res = {
        'Rsquared': rsq_mean,
        'W_Pvalue': mg_W,
        'coefs': pd.concat([coef_mean, tpvalues_mean], axis=1)
      }
      dct['sr_res'] = res
    return dct
  
  def __del__(self) -> None:
    pass


class CDTwoWay:
  '''
  Implementation of the CD test to validate/reject cross-sectional dependence.
  Hypotheses:
  --
  H0: p_{ij} = 0 (No Significant Cross-Sectional Dependence)\n
  H1: p_{ij} != 0 (Valid Cross-Sectional Dependence)\n
  PARAMETERS:
  ----
  - *df*: a Pandas DataFrame containing panel data. Make sure your data is structured in this exact order (by column index):\n
    0 - Spatial units. The data must be homogenous, e.g. only cities, contries, regions, etc.\n
    1 - Temporal units. The data must be homogenous, e.g. only years, months, quarters, etc. \n
    2 - Your target/endogenous variable. The data must not contain NaN values.\n
    3+ - Your exogenous variables. The data must not contain NaN values.\n
  - *level*: The test significance level. Pass an integer, defaults to 5.
  -----
  RETURNS:
  --
  - Prints a string of text via the "verdict" method - the CD-test results.
  '''
  def __init__(self, df: pd.DataFrame, level: int = 5) -> None:
    self.__df = df
    self.__exog = len(self.__df.columns[3:])
    self.__l = []
    self.__alpha = level/100
    for i in range(1, self.__exog+1):
      self.__l.append(f'x{i}')
    self.__df.columns = ['SpUnit', 'time', 'target'] + self.__l
    self.__N = len(self.__df.SpUnit.unique())
    self.__T = len(self.__df.time.unique())
    
  def __resids(self) -> list:
    resids = []
    for unit in self.__df.SpUnit.unique():
      subdf = self.__df[self.__df.SpUnit == unit].copy(deep=True)
      res = sm.OLS(subdf['target'], sm.add_constant(subdf[self.__l])).fit()
      resids.append(res.resid)
    return resids
  
  def __fit(self) -> float:
    corrs = []
    pairs = combinations(self.__resids(), r=2)
    for a, b in pairs:
      cr = np.corrcoef(a, b)[1, 0]
      corrs.append(cr)
    return np.sum(corrs)
  
  def verdict(self) -> None:
    Z = sc.norm()
    CD = self.__fit() * np.sqrt((2*self.__T)/(self.__N*(self.__N-1)))
    pval = 2*min(Z.sf(CD), Z.cdf(CD))
    if pval < self.__alpha:
      print(f'p-value = {pval} < alpha = {self.__alpha}\n There is Significant Cross-Sectional Dependence in your data according to the CD-test. \n Significance level: {self.__alpha*100}%')
    else:
      print(f'p-value = {pval} > alpha = {self.__alpha} There is No Significant Cross-Sectional Dependence in your data according to the CD-test. \n Significance level: {self.__alpha*100}%')


class SlopeHomogeneityF:
  '''
  <h2>The implementation of the F-test for data slope Homogeneity in panel data.</h2>
  <hr>
  <h2>Hypotheses:</h2>
  <hr>
  <ul>
  <li> <em>H0</em>: b_i = b_j given that i != j</li>
  <li><em>H_1</em>: b_i != b_j for at least one such pair of (i, j) that i != j </li>
  </ul>
  <hr>
  <h2>PARAMETERS:</h2>
  <hr>
  <ul>
  <li> <strong>df</strong>: a Pandas DataFrame containing panel data. Make sure your data is structured in this exact order (by column index):
    <ul>
    <li>0 - Spatial units. The data must be homogenous, e.g. only cities, contries, regions, etc.</li>
    <li>1 - Temporal units. The data must be homogenous, e.g. only years, months, quarters, etc.</li>
    <li>2 - Your target/endogenous variable. The data must not contain NaN values.</li>
    <li>3+ - Your exogenous variables. The data must not contain NaN values.</li>
    </ul>
  <li> <strong>level</strong>: Your significance level to test at. Defaults to 5%.</li>
  <li> <strong>intercept</strong>: Specify whether your model has an intercept. Defaults to True.</li>
  <li> <strong>skip_premise</strong>: Set to True if you wish to ignore heteroskedasticity. Not a recommended option, parameter defaults to False</li>
  </ul>
  <hr>
    <h3><em>Note that this test has got certain assumptions, such as error homoskedasticity, N < T.
     The <FTwoWay> instance will check for these conditions to be met, and will yield an error if they are not.</em></h3>
  <hr>
  <h2>RETURNS:</h2>
  <ol>
  <li>Via the instance.verdict() method - prints the result of the F-test.</li>
  </ol>
  '''
  def __init__(self, df: pd.DataFrame, level: int = 5, intercept: bool = True, skip_premise: bool = False) -> None:
    self.__df = df
    self.__l = []
    self.__C = intercept
    self.__alpha = level/100
    self.__k = 0
    self.__skip = skip_premise
    for i, ex in enumerate(self.__df.columns[3:], 1):
      self.__k +=1
      self.__l.append(f'x{i}')
    self.__df.columns = ['SpUnit', 'time', 'target'] + self.__l
    self.__N = len(self.__df.SpUnit.unique())
    self.__T = len(self.__df.time.unique())
    self.__homo_res = het_goldfeldquandt(self.__df['target'], self.__df.iloc[:, 3:])
    self.__verify()
    
  def __verify(self) -> None:
    if self.__N > self.__T:
      raise AssertionError('The F-test for slope Homogeneity assumes that N< T!')
    if self.__homo_res[1] < self.__alpha:
      if not self.__skip:
        raise AssertionError('The Data contains Heteroskedastic errors according to the Goldfeld-Quandt test!')
      else:
        pass
  
  def fit(self) -> float | int:
    USSR = 0
    if self.__C:
      RSSR = sm.OLS(self.__df.iloc[:, 2], sm.add_constant(self.__df.iloc[:, 3:])).fit().ssr
    else:
      RSSR = sm.OLS(self.__df.iloc[:, 2], self.__df.iloc[:, 3:]).fit().ssr
    for unit in self.__df.SpUnit.unique():
      subdf = self.__df[self.__df.SpUnit == unit]
      if self.__C:
        USSR += sm.OLS(subdf.iloc[:, 2], sm.add_constant(subdf.iloc[:, 3:])).fit().ssr
      else:
        USSR += sm.OLS(subdf.iloc[:, 2], subdf.iloc[:, 3:]).fit().ssr
    F = round(self.__N * (self.__T - self.__k - 1) / (self.__k * (self.__N - 1)))* (RSSR - USSR)/USSR
    return F
  
  def verdict(self) -> None:
    f = sc.f(self.__k*(self.__N - 1), self.__N * (self.__T - self.__k - 1))
    pval = f.sf(self.fit())
    if pval < self.__alpha:
      print(f'P-Value: {pval} < Alpha: {self.__alpha}, hence your data does not fit slope homogeneity.')
    else:
      print(f'P-Value: {pval} > Alpha: {self.__alpha}, hence your data fits slope homogeneity.')
    
  def __del__(self) -> None:
    pass
  