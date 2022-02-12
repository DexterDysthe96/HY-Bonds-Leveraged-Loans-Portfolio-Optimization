# Dexter Dysthe and Saharsh Jhunjhunwala
# Professor Carr
# B8328: From Feast to Famine & Back Again: Investing in the Credit Markets through Cycles
# Final Project

from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from matplotlib import pyplot as plt

import numpy as np
import seaborn as sns

sns.set()


def spread_duration(threem_LIBOR, qm, dm, price, years_till_maturity):
    """
    The purpose of the below function is to calculate a proxy for spread duration for floating rate notes.
    We use the formula for modified duration from Fabozzi and Mann's "Introduction to Fixed Income Analytics",
    specifically Ch.12, page 347.

    :param threem_LIBOR:        Three month LIBOR rate. At the time we wrote this, was at 31.5 bps.
    :param qm:                  Quoted margin of a floating rate note, i.e. the spread the coupon pays over LIBOR
    :param dm:                  Discount margin of a floating rate note, i.e. the yield spread over LIBOR (OAS in bond parlance)
    :param price:               The price of the floating rate note.
    :param years_till_maturity: Years till maturity.
                                Note: the below function is not robust to handle non-integer years till maturity
    """
    total_payment_dates = 12 * years_till_maturity
    coupon = threem_LIBOR + qm
    coupon_components = np.array([((coupon / 12) * t)/((1 + (threem_LIBOR + dm) / 12)**(t+1)) for t in
                                  range(1, total_payment_dates + 1)])

    principal_component = (total_payment_dates * 100) / ((1 + (threem_LIBOR + dm)/12)**(total_payment_dates+1))

    spread_dur = (np.sum(coupon_components) + principal_component) / (12 * price)

    return spread_dur / (1 + ((threem_LIBOR + dm) / 12))


def opt_pf(params):
    # Each of the h's correspond to the individual portfolio weights (h is the standard variable for
    # portfolio weights used in academic finance) for the following bonds and loans:
    #       (i) h_m:            Macy Sr Unsec'd Bond (intermediate)
    #       (ii) h_hca:         HCA 1st Lien Bond
    #       (iii) h_nxp:        NXP Interm Sr Unsec'd Green Bond
    #       (iv) h_jane:        JANE Secured Bond
    #       (v) h_rlgy_tl:      RLGY Term Loan
    #       (vi) h_rlgy_unsec:  RLGY Sr Unsec'd Bond
    #       (vii) h_czr_tl:     CZR Term Loan
    #       (viii) h_czr_unsec: CZR Sr Unsec'd Bond
    h_m, h_hca, h_nxp, h_jane, h_rlgy_tl, h_rlgy_unsec, h_czr_tl, h_czr_unsec = params

    # We will successively add to tot_return each of the components coming from our single-names
    tot_return = 0

    # Three month LIBOR rate as of early February
    threem_LIBOR = .003150

    # rf_0 denotes the 5-year treasury rate as of February 4th (Prof Carr listed 1.26% in her email). Our
    # base case assumption is that there will be three 25 bps hikes during 2022, and thus our year end
    # treasury expectation is 201 bps (126 + 75). In our Excel we show how our portfolio performs in a
    # downside case scenario of their being four rate hikes totalling 100 bps.
    rf_0 = 126/100
    rf_t = 201/100

    # In each of the below we include data for starting yield spreads and also our projections for year
    # end spreads. These are our base case assumptions. We assemble our portfolio according to these
    # expectations; in our Excel we show how our portfolio performs in the case of circumstances less
    # desirable than these. We assume the following projections, which are based on those provided by Barclays in
    # their 2022 outlook pieces:
    #       (i)   BBBs: tighten by 15 bps
    #       (ii)  BBs: tighten by 25 bps
    #       (iii) Bs: tighten by 40 bps
    #       (iv)  CCCs: tighten by 65 bps

    # Macy data: duration, starting and projected spread, and yield
    m_duration = 2.1
    m_s0 = 244/100
    m_s1 = (244-25)/100
    m_yield = 3.9

    # Updated total return for M
    tot_return += (m_yield + ((m_s0 - m_s1) + (rf_0 - rf_t))*m_duration)*h_m

    # HCA data: duration, starting and projected spread, and yield
    hca_duration = 6.3
    hca_s0 = 119 / 100
    hca_s1 = (119-15) / 100
    hca_yield = 2.6

    # Updated total return for HCA
    tot_return += (hca_yield + ((hca_s0 - hca_s1) + (rf_0 - rf_t)) * hca_duration) * h_hca

    # NXP data: duration, starting and projected spread, and yield
    nxp_duration = 8.1
    nxp_s0 = 97 / 100
    nxp_s1 = (97-15) / 100
    nxp_yield = 2.4

    # Updated total return for NXP
    tot_return += (nxp_yield + ((nxp_s0 - nxp_s1) + (rf_0 - rf_t)) * nxp_duration) * h_nxp

    # JANE data: duration, starting and projected spread, and yield
    jane_duration = 4.3
    jane_s0 = 275 / 100
    jane_s1 = (275-25) / 100
    jane_yield = 4.1

    # Updated total return for JANE
    tot_return += (jane_yield + ((jane_s0 - jane_s1) + (rf_0 - rf_t)) * jane_duration) * h_jane

    # RLGY Term Loan data: calculated spread duration, starting and projected spread, and yield
    rlgy_tl_duration = spread_duration(threem_LIBOR, 0.0225, 0.0225, 100, 3)
    rlgy_tl_s0 = 225 / 100
    rlgy_tl_s1 = (225-25) / 100
    rlgy_tl_yield = 3.4

    # Updated total return for RLGY term loan
    tot_return += (rlgy_tl_yield + (rlgy_tl_s0 - rlgy_tl_s1) * rlgy_tl_duration) * h_rlgy_tl

    # RLGY Sr Unsec'd Bond data: duration, starting and projected spread, and yield
    rlgy_unsec_duration = 3.5
    rlgy_unsec_s0 = 338 / 100
    rlgy_unsec_s1 = (338-40) / 100
    rlgy_unsec_yield = 4.9

    # Updated total return for RLGY unsec'd bond
    tot_return += (rlgy_unsec_yield + ((rlgy_unsec_s0 - rlgy_unsec_s1) + (rf_0 - rf_t)) * rlgy_unsec_duration) * h_rlgy_unsec

    # CZR Term Loan data: calculated spread duration, starting and projected spread, and yield
    czr_tl_duration = spread_duration(threem_LIBOR, 0.035, 0.0335, 100.25, 3)
    czr_tl_s0 = 335 / 100
    czr_tl_s1 = (335-40) / 100
    czr_tl_yield = 4.5

    # Updated total return for CZR term loan
    tot_return += (czr_tl_yield + (czr_tl_s0 - czr_tl_s1) * czr_tl_duration) * h_czr_tl

    # CZR Sr Unsec'd Bond data: duration, starting and projected spread, and yield
    czr_unsec_duration = 4.2
    czr_unsec_s0 = 275 / 100
    czr_unsec_s1 = (275-65) / 100
    czr_unsec_yield = 4.4

    # Updated total return for CZR unsec'd bond. Total return now includes all the single-name bonds and loans we
    # will invest in
    tot_return += (czr_unsec_yield + ((czr_unsec_s0 - czr_unsec_s1) + (rf_0 - rf_t)) * czr_unsec_duration) * h_czr_unsec

    # Since Python's default optimization engine is a minimizer, we multiply the total return by -1 so that we obtain
    # the portfolio weights that maximize total return (the minimizer of -f(x) equals the maximizer of f(x))
    return -1*tot_return


# ------------------------------------ Duration Times Spread Constraint ------------------------------------ #

three_month_LIBOR = .003150

# Calculate spread duration for the RLGY and CZR term loans
rlgy_tl_duration = spread_duration(three_month_LIBOR, 0.0225, 0.0225, 100, 3)
czr_tl_duration = spread_duration(three_month_LIBOR, 0.035, 0.0335, 100.25, 3)

# Vector of durations and spreads
duration_vector = [2.1, 6.3, 8.1, 4.3, rlgy_tl_duration, 3.5, czr_tl_duration, 4.2]
spread_vector = [244, 119, 97, 275, 225, 338, 335, 275]

# Vector of Duration Times Spread (DTS): see write-up for more details on this measure. Speaking
# briefly, DTS has been shown to be a robust predictor of credit risk inherent in bonds and loans.
# We use DTS as a proxy for credit volatility
duration_times_spread = np.array(duration_vector) * np.array(spread_vector)

# This constraint forces the optimized portfolio to be such that the portfolio's DTS is between
# 600 and 1500. This allows us to take sufficient risk in order to generate attractive returns
# while at the same time placing an upper bound on the amount of spread risk the portfolio can take
linear_constraint_1 = LinearConstraint(duration_times_spread, 600, 1500)


# -------------------------------------- Diversification Constraint -------------------------------------- #
#
# M Debt Inst 2 is rated BB-
# HCA Debt Inst 1 is rated BBB-
# NXP Debt Inst 2 is rated BBB
# JANE Debt Inst 2 is rated BB-
# RLGY Debt Inst 1 is rated BB+
# RLGY Debt Inst 2 is rated B+
# CZR Debt Inst 1 is rated B+
# CZR Debt Inst 3 is rated CCC+

# As discussed on pages 1175-1176 in Ch. 50 of the Bible (Fabozzi's "The Handbook of Fixed Income
# Securities" 8th Edition), maximum position limits imposed to force diversification should not be
# enforced evenly across credit qualities. Thus, the upper bounds we impose below are such that
# smaller values are used for the riskier credit qualities.
#
# Example:  The first entry, (0.05, .30), says that the percentage held of Macy's Sr. Unsec'd bond
#           (intermediate) must be between 5% and 30%. Note that because our single-name credit portfolio
#           constitutes 35% of our total portfolio, these percentages translate to 1.75% and 10.5% of
#           the total portfolio respectively. That is, our investment in Macy must be between 1.75% and
#           10.5% of our total portfolio.
bounds = ((0.05, .30), (.07, .30), (.07, .20), (.07, .30), (.15, .17), (.05, .13), (.05, .175), (.05, .13))


# --------------------------------------- Target Rating Constraint --------------------------------------- #

# We encode ratings according to the following labeling schema:
# BBB-/BBB = 3
# BB+      = 2
# BB-      = 1
# B+       = 0
# B-       = -1
# CCC+     = -2

# Vector of ratings
ratings = [1, 3, 3, 1, 2, 0, 0, -2]

# This constraint forces the portfolio's weighted average rating to be between B+ and BB-. Since we are
# representing ratings as integers, and our portfolio weights are fractional, the weighted average of the
# rating of the portfolio may not be an integer. Thus, we simply demand the value is between 0 and 1, i.e.
# between B+ and BB-. Our rationale for this target rating is largely due to the Barclays 2022 long form
# outlook where they expect B's
linear_constraint_2 = LinearConstraint(np.array(ratings), 0, 1)


# ------------------------------------ Portfolio Weights Constraint ------------------------------------ #

# Recall that the h's correspond to the weights of each of the bonds and loans in our single-name portfolio,
# and as such they must sum to 1.
weights = [1, 1, 1, 1, 1, 1, 1, 1]
linear_constraint_3 = LinearConstraint(np.array(weights), 1, 1)


# ---------------------------------------- Portfolio Optimization --------------------------------------- #

# We now bring all of the above together to construct our optimal single-name credit portfolio

# These are simply seed values for the optimization algorithm. They do not meaningfully affect the output
# of the optimization
dummy_values = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2]
res = minimize(opt_pf, dummy_values, method='trust-constr',
               constraints=[linear_constraint_1, linear_constraint_2, linear_constraint_3],
               options={'verbose': 1}, bounds=bounds)

# Printing values of the portfolio weights. Since our single-name portfolio represents 35% of our total
# portfolio, we multiply the outputs of the optimization by 35 to observe the total percentage each bond
# and loan will comprise.
print("The percentages of different stocks in our portfolio are as follows:")
print((res.x)*35)
print("[M, HCA, NXP, JANE, RLGY TL, RLGY Unsec'd Bond, CZR TL, CZR Unsec'd Bond]")

# Output the single-name credit portfolio's DTS
print("Duration Times Spread: ", round(sum((np.array(duration_vector)*np.array(spread_vector))*res.x), 4))

# Output the single-name credit portfolio's duration
print("Duration: ", round(sum(duration_vector*res.x), 4))

# Output the single-name credit portfolio's total return
print("Total Return: ", round(-opt_pf(res.x), 4))


# ---------------------------------------------- Plotting ---------------------------------------------- #


# ------------------ Plot 1 ------------------ #
#
# Plot explained in detail in write-up

spread_duration_vector = []
returns_list = []
bounds_vector = []
for i in range(750, 1201, 15):
    linear_constraint_1_ub = LinearConstraint(duration_times_spread, 750, i)
    res = minimize(opt_pf, dummy_values, method='trust-constr',
                   constraints=[linear_constraint_1_ub, linear_constraint_2, linear_constraint_3],
                   options={'verbose': 1}, bounds=bounds)
    spread_duration_vector.append(sum((np.array(duration_vector) * np.array(spread_vector)) * res.x))
    returns_list.append(-opt_pf(res.x))
    bounds_vector.append(i)

plt.plot(bounds_vector, returns_list)
plt.title("Optimized Total Return vs Credit Risk")
plt.xlabel("Duration Times Spread Upper Bound")
plt.ylabel("Returns")
plt.show()


# ------------------ Plot 2 ------------------ #
#
# Plot explained in detail in write-up

spread_duration_vector2 = []
returns_list2 = []
bounds_vector2 = []
for i in range(550, 975, 25):
    linear_constraint_1_lb = LinearConstraint(duration_times_spread, i, 1000)
    res = minimize(opt_pf, dummy_values, method='trust-constr',
                   constraints=[linear_constraint_1_lb, linear_constraint_2, linear_constraint_3],
                   options={'verbose': 1}, bounds=bounds)
    spread_duration_vector2.append(sum((np.array(duration_vector) * np.array(spread_vector)) * res.x))
    returns_list2.append(-opt_pf(res.x))
    bounds_vector2.append(i)

plt.plot(bounds_vector2, returns_list2)
plt.title("Optimized Total Return vs Credit Risk")
plt.xlabel("Duration Times Spread Lower Bound")
plt.ylabel("Returns")
plt.show()
