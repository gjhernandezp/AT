# This is the the template algorithm based on an algorithm provided by Leo M that 
# Thomas Wiecki posted Jan 24, 2020 in blog post where the $10K Third-Party
# Challenge: Design a Factor for a Large US Corporate Pension was announced
# https://www.quantopian.com/posts/$10k-third-party-challenge-design-a-factor-for-a-large-us-corporate-pension
# to be used as a template algorithm from challenge.
# 
# The algo uses a documented example from: 
# https://www.quantopian.com/docs/data-reference/ownership_aggregated_insider_transactions
#  
# This template algorithm can not be backtested in the dates range initially 
# provided in the challenge (from January 4, 2014, to August 29, 2018) because 
# the factset ownership Form3AggregatedTrades data is available only up to 2018-04-15. 
# But Thomas Wiecki responded to a post of Joakim Arvidsson (Cream Mongoose)  Jan 25, 2020
# asking "If we want to include the Insiders dataset (two year holdout period), 
# can we use the period January 4, 2014 - January 4, 2018 instead?": "# Good question, 
# yes absolutely. It won't matter for the scoring as we can run it 
# internally on the full IS and OOS period, so the only disadvantage is to you in not 
# being able to see part of the backtest period other users have access to."
# So the backtesting can be done January 4, 2014 - January 4, 2018 if you use the 
# Insiders dataset in your algorithm.
#  
# This template also include the modification 
# constraints=[] —-> constraints=[opt.MaxTurnover(0.2)]
# to obtain daily turnover of 5% to 20%.  Thta is explained 
# in https://www.quantopian.com/tutorials/getting-started#lesson7 
#

from quantopian.algorithm import attach_pipeline, pipeline_output
 
import quantopian.optimize as opt
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.filters import QTradableStocksUS
from quantopian.pipeline.domain import US_EQUITIES
 
# Form 3 transactions
from quantopian.pipeline.data.factset.ownership import Form3AggregatedTrades
# Form 4 and Form 5 transactions
from quantopian.pipeline.data.factset.ownership import Form4and5AggregatedTrades
 
import pandas as pd
import numpy as np
 
def initialize(context):
    """
    Called once at the start of the algorithm.
    """
    # Normally a contest algo uses the default commission and slippage
    # This is unique and only required for this 'mini-contest'
    set_commission(commission.PerShare(cost=0.000, min_trade_cost=0))   
    set_slippage(slippage.FixedSlippage(spread=0))
    
    # Rebalance every day, 1 hour after market open.
    schedule_function(
        rebalance,
        date_rules.every_day(),
        time_rules.market_open(hours=2),
    )
    # Create our dynamic stock selector.
    attach_pipeline(make_pipeline(context), 'pipeline') 
    
    # Record any custom data at the end of each day    
    schedule_function(record_positions, 
                      date_rules.every_day(),
                      time_rules.market_close())
    
    
def create_factor():
    # Base universe set to the QTradableStocksUS
    qtu = QTradableStocksUS()
    # Slice the Form3AggregatedTrades DataSetFamily and Form4and5AggregatedTrades
    # DataSetFamily into DataSets. Here, insider_txns_form3_90d is a DataSet
    # containing insider transaction data for Form 3 over the past 90 calendar
    # days, and insider_txns_form4and5_90d is a DataSet containing insider
    # transaction data for Forms 4 and 5 over the past 90 calendar days. We only
    # include non-derivative ownership (derivative_holdings is False).
    insider_txns_form3_90d = Form3AggregatedTrades.slice(False, 90)
    insider_txns_form4and5_90d = Form4and5AggregatedTrades.slice(False, 90)
    # From each DataSet, extract the number of unique buyers and unique sellers.
    # We do not need to include unique sellers using Form 3, because Form 3 is
    # an initial ownership filing, and so there are no sellers using Form 3.
    unique_filers_form3_90d = insider_txns_form3_90d.num_unique_filers.latest
    unique_buyers_form4and5_90d = insider_txns_form4and5_90d.num_unique_buyers.latest
    unique_sellers_form4and5_90d = insider_txns_form4and5_90d.num_unique_sellers.latest
    # Sum the unique buyers from each form together.
    unique_buyers_90d = unique_filers_form3_90d + unique_buyers_form4and5_90d
    unique_sellers_90d = unique_sellers_form4and5_90d
    # Compute the fractions of insiders buying and selling.
    frac_insiders_buying_90d = unique_buyers_90d / (unique_buyers_90d + unique_sellers_90d)
    frac_insiders_selling_90d = unique_sellers_90d / (unique_buyers_90d + unique_sellers_90d)
    
    # compute factor as buying-selling rank zscores
    alpha_factor = frac_insiders_buying_90d - frac_insiders_selling_90d
    
    screen = qtu & ~alpha_factor.isnull() & alpha_factor.isfinite()
    
    return alpha_factor, screen
 
def make_pipeline(context):  
    alpha_factor, screen = create_factor()
    
    # Winsorize to remove extreme outliers
    alpha_winsorized = alpha_factor.winsorize(min_percentile=0.02,
                                              max_percentile=0.98,
                                              mask=screen)
    
    # Zscore and rank to get long and short (positive and negative) alphas to use as weights
    alpha_rank = alpha_winsorized.rank().zscore()
    
    return Pipeline(columns={'alpha_factor': alpha_rank}, 
                    screen=screen, domain=US_EQUITIES)
    
 
def rebalance(context, data): 
    # Get the alpha factor data from the pipeline output
    output = pipeline_output('pipeline')
    alpha_factor = output.alpha_factor
    log.info(alpha_factor)
    # Weight securities by their alpha factor
    # Divide by the abs of total weight to create a leverage of 1
    weights = alpha_factor / alpha_factor.abs().sum() 
   
    # Must use TargetWeights as an objective
    order_optimal_portfolio(
        objective=opt.TargetWeights(weights),
        constraints=[opt.MaxTurnover(0.2)],
    )
 
    
def record_positions(context, data):
    pos = pd.Series()
    for position in context.portfolio.positions.values():
        pos.loc[position.sid] = position.amount
        
    pos /= pos.abs().sum()
    
    # Show quantiles of the daily holdings distribution
    # to show if weights are being squashed to equal weight
    # or whether they have a nice range of sensitivity.
    quantiles = pos.quantile([.05, .25, .5, .75, .95]) * 100
    record(q05=quantiles[.05])
    record(q25=quantiles[.25])
    record(q50=quantiles[.5])
    record(q75=quantiles[.75])
    record(q95=quantiles[.95])