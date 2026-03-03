import numpy as np
import pandas as pd
import statsmodels.api as sm
from itertools import product
import scipy.stats as sc
from math import floor
import importlib.resources as resources
from statsmodels.regression.mixed_linear_model import MixedLM

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
  -----------------------------
  RETURNS:
  --
  via instance.verdict() -> None: Prints the verdict of the test based on the parameters set by the user.
  '''
  def __init__(self, df: pd.DataFrame, T: int, N: int, trend: bool =  False, poly_trend: int = 1, intercept: bool = False, n_lags: int = 2, level: int = 5) -> None:
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
  -----
  - *data*: A pandas DataFrame. Make sure that all items are introduced in this exact order by column index:\n
    Make sure no columns contain Nan Values!
    0 - your spatial unit. The data must be homogenous, e.g. only contries, regions, etc.
    1 - your temporal window per panel. The data must be homogenous, e.g. only years, months, etc.
    2 - Your target variable.
    3+ - your exogenous variables.
  - *level*: the statistical test significance level. 
  ---------
  METHODOLOGY:
  ------
  - The Test prefers an MLE estimator for RE (Random Effects), 
    which can be disrupted if the difference between FE and RE are considerably small. 
    The test will opt for MLE estimation if sigma_u computed for GLS is greater than 0.
------
  RETURNS:
  -
  - Via instance.verdict() prints the verdict of the Hausman test according to the given parameters.
  '''
  def __init__(self, data: pd.DataFrame, level: int = 5) -> None:
    self.__df = data
    self.__exog = len(data.columns[3:])
    self.__l =[]
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
    re = re.set_index(['SpUnit', 'time'])
    means = re.groupby('SpUnit')[['target'] + self.__l].mean()
    sigma2 = np.sum((sm.OLS(means.iloc[:, 0], sm.add_constant(means.iloc[:, 1:])).fit()).resid**2) / (self.__N - self.__exog - 1)
    sigma_u = sigma2 - w_err/self.__T
    if sigma_u <= 0:
      sigma_u = 0
    elif sigma_u > 0:
      self.__RE = 'MLE'
      return None
    theta = 1 - np.sqrt(w_err/(w_err + self.__T * sigma_u))
    temp = re.merge(means, on='SpUnit', suffixes=('', '_avg'))
    re_df = pd.DataFrame()
    for name in self.__df.columns[2:]:
      re_df.loc[:, f'{name}_norm'] = temp.loc[:, name] - theta * temp.loc[:, f'{name}_avg']
    return re_df
      
      
  def estimate(self) -> float:
    fe_df = self.build_FE()
    Chi = 0
    fe_res = sm.OLS(fe_df.iloc[:, 0], sm.add_constant(fe_df.iloc[:, 1:])).fit()
    re_df = self.build_GLS(fe_res.mse_resid)
    if re_df is not None:
      re_res = sm.OLS(re_df.iloc[:, 0], sm.add_constant(re_df.iloc[:, 1:])).fit()
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




