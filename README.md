Portfolio Investment Thesis
Based on our view of the high yield market and interest rate environment, we recommend a portfolio that reduces duration and overweights allocations to individual credit securities with accretive risk-reward trade-offs.  We believe that rising interest rates will have a greater impact on returns in the next year than changes in spreads or defaults, and our portfolio will benefit from lower duration than the high yield index.  We also expect to outperform the high yield index through allocations to individual securities with superior risk-reward profiles that are not fully reflected in current market pricing (i.e., downside protection or unrealized upside).
Our base case total return expectation for our portfolio is 4.09% compared to 2.88% for the benchmark high yield index, leading to an excess return of 85 bps.  Excluding our 5% allocation to equities – which increases total return, though adds to risk – we still expect our credit-only portfolio total return to exceed the high yield index by 16%, constituting an excess return of 46 bps (3.34% vs. 2.88%, respectively).  See the accompanying portfolio allocation and total return excel for additional detail.  Also, through individual security selection, we believe we have decreased our portfolio’s risk compared to the high yield index, as evidenced by our downside case total return expectation for our portfolio of 0.20% compared to -0.35% for the high yield index, producing an excess return of 55 bps. Excluding the equity bucket, our portfolio returns 0.45%, amounting to an excess return of 81 bps in a downside market environment.

Key Opportunities and Risks
	We expect the key pillars of our investment thesis – shorter overall duration and overweight allocation to quality securities (companies and capital structure) – to somewhat mitigate interest rate risk and loss default potential relative to the high yield index.  However, there are three key risk areas to our proposed allocation that could lead to underperformance:
Interest rates do not increase as much as we anticipate: factors that cause the Fed to postpone anticipated rate hikes would reduce the benefit of holding a shorter duration portfolio
Incorrect assessment of price / value discrepancies: our bottoms-up analyses could have overestimated upside opportunity for bonds on the path to investment grade (i.e., Macy’s) 
Default rates stay low across credit qualities: low default rates reduce the benefit of allocations to lower risk securities (i.e., HCA) relative to higher-yielding bonds within the high yield index
Ultimately, we believe our portfolio should outperform due to optimized risk-reward from individual security selection, including the small equity allocation that compliments the lower risk profile of our credit portfolio relative to the high yield index.

Portfolio Composition and Methodology
	We recommend 60% allocation to indices (45% HY index, 10% CCC index, and 5% bank loan index), 5% allocation to equities (2.5% HCA common stock and 2.5% NXP common stock), and 35% allocation to individual credit securities as follows:
Caesars (10.7%): term loan at CRC (6.1%) and senior unsecured bond at CZR (4.5%)
Realogy (10.5%): term loan (5.9%) and senior unsecured bond (4.5%)
Macy’s (6.5%): intermediate senior unsecured bond
Jane Street (2.5%): secured bond
HCA (2.5%): first lien bond
NXP (2.5%): intermediate senior unsecured green bond
Overall, we used a four-step process to determine our portfolio composition.  First, we articulated our top-down investment themes.  Second, using those themes, we conducted bottom-up analyses of the individual company securities to assess which companies we wanted to invest in and select the best securities in each company’s capital structure.  Then, using our chosen indices and individual securities, we ran a quantitative strategy to optimize expected total return in our base case scenario.  The quantitative approach used a simple algorithm for portfolio allocation to maximize total return within several parameters using the following constraints: 
Duration Times Spread (DTS): In 2007, in joint work by Lehman Brothers and Robeco Asset Management researchers Arik Ben Dor, Lev Dynkin, Jay Hyman, Patrick Houweling, Erik van Leeuwen, and Olaf Penninga, the paper “Duration Times Spread'' was published in The Journal of Portfolio Management. The paper demonstrates that spread duration, a historically commonly used measure of credit risk and spread volatility, was inadequate for capturing the nuances of price volatility for bonds and loans. Whereas spread duration measures the sensitivity of a portfolio to a parallel shift in spreads – that is, an absolute change in spreads – the paper shows that DTS captures relative changes in spread, thus taking into account that credits trading at wider spreads are inherently riskier assets. One of the primary attractive features of DTS is that it is a robust predictor of future credit volatility when compared with other measures. Because of this, instead of constraining the average spread our single-name credit portfolio can take, we limit the portfolio’s DTS exposure. To see how this constraint is implemented, please see lines 157 to 177 in the Python code displayed in Exhibit A below.
Diversification: As discussed in Fabozzi's "The Handbook of Fixed Income Securities" 8th Edition (re: pages 1175-1176 in Ch. 50), position limits imposed to force diversification should not be enforced evenly across credit qualities. Thus, the upper bounds we impose on our single name securities are such that our portfolio is constrained to hold riskier credit qualities in lower proportions compared to relatively safer assets, and lower bounds help ensure that at least a certain fraction of the portfolio is invested in safer securities, even ones with low yield and low spread-duration. This helps ensure diversification and elimination of unsystematic risk. Please refer to line 201 in the Python code presented in Exhibit A for the bounds assigned to our single name holdings.
Target Credit Quality: This constraint forces the portfolio's weighted average rating to be between B+ and BB-. Our rationale for this target rating is largely due to the Barclays 2022 long form outlook where they express their expectation that B's will offer a superior relative risk-return profile compared with other parts of the HY bond market. To simplify the implementation of this constraint, we have assigned each rating an integer value. Since we are representing ratings as integers, and our portfolio weights are fractional, the weighted average rating of the portfolio may not be an integer. Thus, we simply require that this integer value be between 0 and 1, i.e. between B+ and BB-. Please refer to lines 206 through 212 in the Python code segment presented in Exhibit A for the integer values assigned to each credit quality.
 The code underpinning our quantitative strategy is outlined below in Exhibit A. Finally, we applied equity allocations based on our assessment that (i) there is upside in HCA and NXP stock and (ii) a small equity allocation would add some diversification benefits.
	As discussed in the portfolio investment thesis, rising interest rates will place pressure on bond returns, but we still expect lower default rates and select opportunities for spread tightening.  As a result, we do not recommend any allocation to cash.  Additionally, since our mandate will have us benchmarked against the high yield index, we chose to begin with a 45% allocation to the HY index to somewhat anchor the portfolio return.  We also decided to allocate 10% to the CCC index for prudent exposure to higher yielding assets in a diversified and opportunistic bucket that outperformed the market significantly during 2021. Finally, we chose to invest 5% in the leveraged loan index to add diversified exposure to loans, which could stand to benefit from rate hikes due to the floating rate structure.  Overall, we felt that 60% of our portfolio in indices (with the majority in the HY index) would provide significant diversification benefits while centering our portfolio around the benchmark index with opportunities for upside through bottoms-up individual security analyses.

Individual Company and Security Analyses
Caesars (CZR)
Caesars represents our largest single company allocation with investments in the term loan (CRC) and senior unsecured bond (CZR).  We view the term loan (CRC) as an opportunity to invest in a note with strong risk-reward given 4.4% yield (with LIBOR forward curve) and meaningful physical asset protection as the loan sits closest to the legacy CZR assets.  We also like the senior unsecured bond, which has a 5.625% coupon and intermediate duration.  We believe this note has upside as its rating could improve based on the company’s outlook.  We thought both of these securities were superior to the first lien bond, which was furthest away from the CZR assets and trading above its July 2022 call price.
Our thesis for both notes is supported by positive underlying tailwinds for Caesars following its merger with Eldorado.  In particular, we like Caesars’ scale post-merger, efforts to simplify its cash structure, physical asset protection, progress on cost cutting initiatives, and gaming industry recovery.   The risks posed by COVID-19 have started to subside, which has helped stage a rebound for the gaming sector (both gaming and non-gaming/entertainment revenue). This trend is reflected in the lift in occupancy rates in Caesars’ Vegas strip. Though the gaming industry is not fully recovered, we believe it is trending in the right direction.  In addition, the Eldorado merger creates the largest U.S. casino operator with a significant presence in Vegas and significant cost synergies.  Lastly, Caesars is generating strong free cash flow relative to historical performance and has retail assets as collateral for secured creditors.
Realogy (RLGY)
	We elected to include both of Realogy’s credit securities in our portfolio - the term loan and the senior unsecured bond.  For the term loan, we felt the first lien on RLGY’s assets is not particularly meaningful given it is an asset-light business, but the priority position in the capital structure could provide protection in a housing downturn.  Additionally, the term loan has a floating rate with no LIBOR floor, which allows us to participate in the upside associated with a rising rate environment.  
For the senior unsecured bond, the lack of collateral presents a risk, but at only 0.8x secured leverage we thought the bondholders would be well-positioned to be the fulcrum security and benefit from the potential upside.  In addition, we view the +150bps of yield versus the term loan as favorable given no meaningful asset protection for either.  Finally, the company is currently at a much more reasonable leverage profile (3.0x Debt / EBITDA) relative to post Apollo/TPG-LBO.
Our main concern from an operational and industry standpoint was the cyclicality of the housing industry.  However, the downside risk is somewhat mitigated by the lower leverage profile and variable cost structure.  The difficulty in assessing the trajectory of the housing market also contributed to our decision to hold both securities, allowing us to have ultimate downside protection (first lien), while also positioning for the fulcrum security (bond).  Lastly, most housing market research suggests reasonable underlying fundamentals, which should reduce the one-year portfolio risk.
Macy’s (M)
	We chose to invest in Macy’s intermediate senior unsecured bond.  We felt the intermediate bond, Macy’s most recently issued note, had the preferred risk-reward trade-off in the capital structure and could benefit as spreads tighten on Macy’s migration back to investment grade.  In comparison, the shortest maturity note is already priced as if it will be paid at par and the longest maturity note is highly rate-sensitive with long duration.  Given our perspective on interest rates in 2022, this made the longest maturity note fairly unattractive.  Additionally, the intermediate bond has greater yield than the HY index and may have lower default risk given Macy’s trajectory to IG - therefore making it accretive to our portfolio.  We are also comfortable with the senior unsecured position in the capital structure because Macy’s has significant physical assets to collateralize the priority lenders ahead of bondholders.
	In our operational and industry analysis of Macy’s we were concerned by overall department store trends, but encouraged by Macy’s evidence of margin improvement, emerging ecommerce business, and reasonable overall leverage profile.  This combination of factors, coupled with physical assets, made us comfortable with the credit.  However, from an equity perspective, we viewed the lack of growth in the core department store business as outsized risk and were more comfortable holding the bonds.
Jane Street (JANE)
	We elected to allocate capital to JANE’s secured bond.  We would invest in Jane Street’s bonds due to the management team’s focus on risk management and track record of talent retention.  Looking at the capital structure, we do not see much value in JANE’s asset collateral (given trade debts), so we did not feel that the term loan added much additional protection to the pari passu secured bond.  In addition, we value the call protection of the secured bond relative to the term loan.  We perceive risk in the floating rate nature of the term loan given the rising interest rate environment, coupled with Jane Street’s strong financial performance.  This could create a dynamic where the term loan is called in the near-term, whereas the bond has protections.
	In terms of operational and industry considerations we feel that the continued electronification of the trading process for credit securities will widen the universe of products Jane Street can make markets in.  Nonetheless, this disruption also poses risks as it has the ability to take away market share from Jane Street by attracting competitors due to further improving profitability. This increased competition might force Jane Street to loosen its risk management standards, a substantial risk outlined in the Moodys’ assessment, thus increasing risk for its lenders. 
We feel that Jane Street’s arsenal of traditional traders in addition to their quants and software engineers effectively mitigates any potential “black box” risk. Unlike many other recent entrants in the market making universe – which rely almost entirely on a purely technology and algorithm-driven approach to trading – Jane Street stands with few others at the forefront of the intersection of quant and fundamental approaches to market making. 
HCA (HCA)
	We propose an investment in HCA’s first lien bond.  We believe this is the best credit in HCA’s capital structure given first priority for collateral at relatively similar yield and shorter duration than the senior unsecured bond.  We did not feel like the long duration first lien bond had enough upside to warrant the duration risk in the rising interest rate environment.  
	We view the HCA bond as attractive downside protection in our portfolio given the non-cyclical nature of hospitals and the company’s strong cash flow profile.  While COVID has placed pressure on hospitals, HCA continues to produce positive cash flows amidst increasing patient volume.  We also reviewed political / legislative risk and were able to get comfortable with HCA’s ability in previous “cycles” of policy change to maintain double digit EBITDA margins and cover interest payments, particularly at today’s lower leverage ratios (compared to historicals).
NXP (NXP)
	We recommend an investment in NXP’s intermediate senior unsecured green bond.  This bond represents the best risk-adjusted return within the capital structure because of its tax advantages, maturity alignment with NXP’s long-term contract visibility, intermediate duration, and ESG angle. In comparison to the HCA secured debt, which shares the same rating, the NXP unsecured debt has less asset protection.  However, the potential return on NXP’s credit compares favorably to HCA’s secured debt when taking into account tax advantages: NXP’s tax-exempt green bond’s yield is comparable to a pre-tax yield over 3% for a taxable security (i.e., HCA).
	We also closely evaluated the trajectory of the semiconductor industry in our assessment of NXP and its credit securities. While the semiconductor industry is highly cyclical, we believe there are strong fundamental shifts in the market that are accelerating the visible long-term demand for chips and have driven the industry into a massive chip shortage. The industry is currently benefiting from the shift to turn everything from refrigerators to cars into “connected smart devices,” which drove over 20% industry growth in 2021.  Over the next 7 years, the industry is expected to grow at a 9% CAGR from $452bn to $800bn+ (per Fortune Business Insights). Given the long-term industry dynamics, current shortage challenges, and need for innovation, we believe that it is reasonable for semiconductor companies to raise debt to create more efficient chips and better manufacturing techniques to increase total production capacity.  We are also comfortable with NXP because of its prudent total leverage profile (1.6x net debt to EBITDA), strong EBITDA margins (37%), and free cash flow conversion (66%+ of EBITDA).
	Finally, we think the green label is an important consideration for us to actively include in our portfolio.  NXP has issued over $2 billion in green bonds with the goal of improving the energy efficiency of chips, as well as, reducing the carbon footprint of its manufacturing process.  We believe these efforts should produce financial and environmental returns, though accurately measuring the net positive impact of these investments is particularly challenging.  For our portfolio, we believe the NXP bond helps us improve the ESG metrics of our portfolio, which is an important consideration for many Limited Partners.

Equity Allocations
	We recommend 5% allocation of our portfolio to equities (2.5% HCA common stock and 2.5% NXP common stock).  We felt that our individual credit security selections overall potentially provided lower risk than the average risk in the HY index, so warranted small exposure to equities.  The equity allocations also provide additional diversification benefits.  Lastly, the markets have presented a potentially advantageous buying opportunity due to the recent pullback in stock prices.
	We chose to invest in the HCA common stock because of the company’s non-cyclical nature and track record of meaningful share repurchase and shareholder dividend programs.  We like the moat provided by HCA operating the largest network of hospitals in the U.S. with dominant market share in many of its geographic focus areas.  We think HCA’s cost advantages will persist and that policy / legislative changes in the healthcare space have a long lead time so do not pose a significant risk in the near-term.  HCA is a “value” stock that we believe has upside.
	We also chose to invest in NXP given the company’s operating margins, strong historical revenue growth, and dividend / share buyback program.  While NXP is trading at ~29x P/E, which is in line with its publicly traded peers, we believe its size and diversified end markets make it a more attractive investment opportunity.  Finally, as discussed above, we believe the massive disconnect between chip demand and supply provides NXP with strong medium-term revenue visibility.  
