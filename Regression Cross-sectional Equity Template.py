"""
Regression Cross-sectional Equity Template 
German Hernandez

This algorithm uses a linear regression predictor of the return PRED_N_FORWARD_DAYS latter to be used as combined_factor for the [Cross-sectional Equity Template] (https://www.quantopian.com/algorithms/5d14d0dace2ee337d7289254) included in the examples. 

I use the segments of code  form the great posts [Machine Learning on Quantopian Part 3: Building an Algorithm - Thomas Wiecki](https://www.quantopian.com/posts/machine-learning-on-quantopian-part-3-building-an-algorithm)  and [How to Leverage the Pipeline to Conduct Machine Learning in the IDE –  Jim Obreen](https://www.quantopian.com/posts/how-to-leverage-the-pipeline-to-conduct-machine-learning-in-the-ide) to build the ML(CustomFactor) and the pipeline. 
"""

import quantopian.algorithm as algo
import quantopian.optimize as opt
from quantopian.pipeline import Pipeline, CustomFactor
from quantopian.pipeline.factors import SimpleMovingAverage

from quantopian.pipeline.filters import QTradableStocksUS
from quantopian.pipeline.experimental import risk_loading_pipeline

from quantopian.pipeline.data.psychsignal import stocktwits
from quantopian.pipeline.data import Fundamentals
from quantopian.pipeline.factors import Returns

from collections import OrderedDict

from quantopian.pipeline.data.builtin import USEquityPricing

# The basics
import pandas as pd
import numpy as np

# SKLearn :)
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.preprocessing import Imputer, StandardScaler



##################################################
# Globals 
##################################################

# Constraint Parameters
MAX_GROSS_LEVERAGE = 1.0
TOTAL_POSITIONS = 600

# train on returns over N days into the future
PRED_N_FORWARD_DAYS = 5
TRAINING_WINDOW_DAYS = 30
PERCENT = 0.05
MAX_SHORT_POSITION_SIZE = 2.0 / TOTAL_POSITIONS
MAX_LONG_POSITION_SIZE = 2.0 / TOTAL_POSITIONS
universe = QTradableStocksUS()

def initialize(context):
    """
    A core function called automatically once at the beginning of a backtest.

    Use this function for initializing state or other bookkeeping.

    Parameters
    ----------
    context : AlgorithmContext
        An object that can be used to store state that you want to maintain in 
        your algorithm. context is automatically passed to initialize, 
        before_trading_start, handle_data, and any functions run via schedule_function.
        context provides the portfolio attribute, which can be used to retrieve information 
        about current positions.
    """
    
    algo.attach_pipeline(make_pipeline(), 'long_short_equity_template')

    # Attach the pipeline for the risk model factors that we
    # want to neutralize in the optimization step. The 'risk_factors' string is 
    # used to retrieve the output of the pipeline in before_trading_start below.
    algo.attach_pipeline(risk_loading_pipeline(), 'risk_factors')

    # Schedule our rebalance function
    algo.schedule_function(func=rebalance,
                           date_rule=algo.date_rules.week_start(),
                           time_rule=algo.time_rules.market_open(hours=0, minutes=30),
                           half_days=True)

    # Record our portfolio variables at the end of day
    algo.schedule_function(func=record_vars,
                           date_rule=algo.date_rules.every_day(),
                           time_rule=algo.time_rules.market_close(),
                           half_days=True)

class ML(CustomFactor):
    """
    """
    train_on_weekday = 1

    def __init__(self, *args, **kwargs):
        CustomFactor.__init__(self, *args, **kwargs)

        self._regressor  = LinearRegression()
        self._trained = False

    def _compute(self, *args, **kwargs):
        ret = CustomFactor._compute(self, *args, **kwargs)
        return ret

    def compute(self, today, assets, out, *inputs): #compute(self, today, assets, out, returns, *inputs):
        # inputs has the Returns n_forward_days+1 and the 3 alpha factors from Cross-sectional Equity Template
        #  inputs[0]: (Returns)
        # [[1, 3, 2], # factor 1 rankings of day t-1 for 3 stocks
        #  [3, 2, 1]] # factor 1 rankings of day t for 3 stocks
        # inputs[1]:
        # [[2, 3, 1], # factor 2 rankings of day t-1 for 3 stocks
        #  [1, 2, 3]] # factor 2 rankings of day t for 3 stocks
        
        
        print('preprocessing starting on: {}'.format(today))
        
        columns = ['Returns', 'sentiment_score', 'quality', 'value']
        
        regressor = self._regressor
            
        inputs = OrderedDict([(columns[i] , pd.DataFrame(inputs[i]).fillna(method='ffill',axis=1).fillna(method='bfill',axis=1)) for i in range(len(inputs))]) # bring in data with some null handling.
            
        y = inputs['Returns']. shift(-PRED_N_FORWARD_DAYS).dropna(axis=0,how='all').stack(dropna=False)
      
        num_secs = len(inputs['Returns'].columns)
 
        x = pd.concat([df.stack(dropna=False) for df in inputs.values()], axis=1)
        x.columns = columns
                
        x = Imputer(strategy='median',axis=1).fit_transform(x) # fill nulls.         
        y = np.ravel(Imputer(strategy='median',axis=1).fit_transform(y)) # fill nulls.
        scaler = StandardScaler()
        x = scaler.fit_transform(x) # demean and normalize
        x_t =x[-num_secs:,:]
        x = x[:-num_secs*(PRED_N_FORWARD_DAYS),:]         
        
        if (today.weekday() == self.train_on_weekday) or not self._trained:
            
            print('training model for window starting on: {}'.format(today))
            
            regressor.fit(x, y)
           
            print('score',regressor.score(x, y))     
        
            self._trained = True
            
        out[:] = regressor.predict(x_t)
        
def make_pipeline():
    """
    A function that creates and returns our pipeline.

    We break this piece of logic out into its own function to make it easier to
    test and modify in isolation. In particular, this function can be
    copy/pasted into research and run by itself.

    Returns
    -------
    pipe : Pipeline
        Represents computation we would like to perform on the assets that make
        it through the pipeline screen.
    """
    
    value = Fundamentals.ebit.latest / Fundamentals.enterprise_value.latest
    quality = Fundamentals.roe.latest
    sentiment_score = SimpleMovingAverage(
        inputs=[stocktwits.bull_minus_bear],
        window_length=3,
    )
    
    input_columns= {
            'value': value,
            'quality': quality,
            'sentiment_score': sentiment_score,
    }   
    
    predictor_columns = OrderedDict()
    
    predictor_columns['Returns'] = Returns(
        inputs=(USEquityPricing.open,),
        mask=universe, window_length=PRED_N_FORWARD_DAYS + 1,
    )

    # rank all the factors and put them after returns
    predictor_columns.update({
        k:v.winsorize(min_percentile=PERCENT, max_percentile=1.0-PERCENT).zscore()  for k, v in input_columns.items()
    })
    # v.rank(mask=universe)
    
    print predictor_columns.keys()
    
    # Create our ML pipeline factor. The window_length will control how much
    # lookback the passed in data will have.
    combined_factor = ML(
        inputs=predictor_columns.values(),
        window_length=TRAINING_WINDOW_DAYS,
        mask=universe,
    )
    
    # Build Filters representing the top and bottom baskets of stocks by our
    # combined ranking system. We'll use these as our tradeable universe each
    # day.
    longs = combined_factor.top(TOTAL_POSITIONS//2, )#mask=universe)
    shorts = combined_factor.bottom(TOTAL_POSITIONS//2,)# mask=universe)

    # The final output of our pipeline should only include
    # the top/bottom 300 stocks by our criteria
    long_short_screen = (longs | shorts)

    # Create pipeline
    pipe = Pipeline(
        columns={
            'longs': longs,
            'shorts': shorts,
            'combined_factor': combined_factor
        },
        screen=long_short_screen
    )
    return pipe


def before_trading_start(context, data):
    """
    Optional core function called automatically before the open of each market day.

    Parameters
    ----------
    context : AlgorithmContext
        See description above.
    data : BarData
        An object that provides methods to get price and volume data, check
        whether a security exists, and check the last time a security traded.
    """
    # Call algo.pipeline_output to get the output
    # Note: this is a dataframe where the index is the SIDs for all
    # securities to pass my screen and the columns are the factors
    # added to the pipeline object above
    context.pipeline_data = algo.pipeline_output('long_short_equity_template')

    # This dataframe will contain all of our risk loadings
    context.risk_loadings = algo.pipeline_output('risk_factors')


def record_vars(context, data):
    """
    A function scheduled to run every day at market close in order to record
    strategy information.

    Parameters
    ----------
    context : AlgorithmContext
        See description above.
    data : BarData
        See description above.
    """
    # Plot the number of positions over time.
    algo.record(num_positions=len(context.portfolio.positions))


# Called at the start of every month in order to rebalance
# the longs and shorts lists
def rebalance(context, data):
    """
    A function scheduled to run once every Monday at 10AM ET in order to
    rebalance the longs and shorts lists.

    Parameters
    ----------
    context : AlgorithmContext
        See description above.
    data : BarData
        See description above.
    """
    # Retrieve pipeline output
    pipeline_data = context.pipeline_data

    risk_loadings = context.risk_loadings

    # Here we define our objective for the Optimize API. We have
    # selected MaximizeAlpha because we believe our combined factor
    # ranking to be proportional to expected returns. This routine
    # will optimize the expected return of our algorithm, going
    # long on the highest expected return and short on the lowest.
    objective = opt.MaximizeAlpha(pipeline_data.combined_factor)

    # Define the list of constraints
    constraints = []
    # Constrain our maximum gross leverage
    constraints.append(opt.MaxGrossExposure(MAX_GROSS_LEVERAGE))

    # Require our algorithm to remain dollar neutral
    constraints.append(opt.DollarNeutral())

    # Add the RiskModelExposure constraint to make use of the
    # default risk model constraints
    neutralize_risk_factors = opt.experimental.RiskModelExposure(
        risk_model_loadings=risk_loadings,
        version=0
    )
    constraints.append(neutralize_risk_factors)

    # With this constraint we enforce that no position can make up
    # greater than MAX_SHORT_POSITION_SIZE on the short side and
    # no greater than MAX_LONG_POSITION_SIZE on the long side. This
    # ensures that we do not overly concentrate our portfolio in
    # one security or a small subset of securities.
    constraints.append(
        opt.PositionConcentration.with_equal_bounds(
            min=-MAX_SHORT_POSITION_SIZE,
            max=MAX_LONG_POSITION_SIZE
        ))

    # Put together all the pieces we defined above by passing
    # them into the algo.order_optimal_portfolio function. This handles
    # all of our ordering logic, assigning appropriate weights
    # to the securities in our universe to maximize our alpha with
    # respect to the given constraints.
    algo.order_optimal_portfolio(
        objective=objective,
        constraints=constraints
    )