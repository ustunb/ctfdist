# ctfdist

`ctfdist` is a package to repair classification models with performance disparities between demographic subgroups.

Highlights:

- works with any binary classification model
- repair does not require training a new classification model
- can repair performance disparities for common fairness metrics (differences in TPR, FPR, Equal Opportunity)
- can repair performance disparities for multiple subgroups

### Paper

[Repairing without Retraining: Avoiding Disparate Impact with Counterfactual Distributions](https://arxiv.org/pdf/1901.10501.pdf) 
by Hao Wang, Berk Ustun, and Flavio Clamon

 ```

@ARTICLE{wang2019ctf,
       author = {{Wang}, Hao and {Ustun}, Berk and {Calmon}, Flavio P.},
        title = "{Repairing without Retraining: Avoiding Disparate Impact with Counterfactual Distributions}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Machine Learning, Computer Science - Computers and Society, Computer Science - Information Theory, Statistics - Machine Learning},
         year = "2019",
          eid = {arXiv:1901.10501},
        pages = {arXiv:1901.10501},
archivePrefix = {arXiv},
       eprint = {1901.10501},
}
```


### Package Installation

The minimum requirements are:

- Python 3.5+ 
- CPLEX 12.6+
 
The code may still work with older versions of Python and CPLEX, but this will not be tested or supported. 

#### CPLEX 

CPLEX is cross-platform commercial optimization tool with a Python API. It is freely available to students and faculty members at accredited institutions. To get CPLEX:

1. Register for [IBM OnTheHub](https://ibm.onthehub.com/WebStore/Account/VerifyEmailDomain.aspx)
2. Download the *IBM ILOG CPLEX Optimization Studio* from the [software catalog](https://ibm.onthehub.com/WebStore/ProductSearchOfferingList.aspx?srch=CPLEX)
3. Install the CPLEX Optimization Studio.
4. Setup the CPLEX Python API [as described here](https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/set_up/Python_setup.html).

If you have problems installing CPLEX, check the [CPLEX user manual](http://www-01.ibm.com/support/knowledgecenter/SSSA5P/welcome) or the [CPLEX forums](https://www.ibm.com/developerworks/community/forums/html/forum?id=11111111-0000-0000-0000-000000002059).

#### Development Timeline

*** This package is in active development; code will change substantially with each commit***

- scikit-learn API conventions
- refactoring for future development
- integration of open-source LP solver to create preprocessor
     


