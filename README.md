# ctfdist

`ctfdist` is a package to repair classification models that perform differently on the basis of protected attributes (e.g., gender, age group).

**Highlights**

- works with any binary classification model 
- repairs disparities without training a new model
- address popular fairness criteria (e.g., group-specific differences in TPR, FPR, equalized odds)
- can be used to address disparities for more than 2 protected groups

**Paper**

[Repairing without Retraining: Avoiding Disparate Impact with Counterfactual Distributions](https://arxiv.org/pdf/1901.10501.pdf)    
    Hao Wang, Berk Ustun, and Flavio P. Calmon

 ```
@article{wang2019ctf,
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

## Installation

Minimum requirements: 

- Python 3.6+
- CPLEX 12.6+,
 
The code may work with older versions of Python and CPLEX, but this will not be tested or supported. 

#### Getting CPLEX 

CPLEX is cross-platform commercial optimization tool with a Python API. It is freely available to students and faculty members at accredited institutions. To get CPLEX:

1. Register for [IBM OnTheHub](https://ibm.onthehub.com/WebStore/Account/VerifyEmailDomain.aspx)
2. Download the *IBM ILOG CPLEX Optimization Studio* from the [software catalog](https://ibm.onthehub.com/WebStore/ProductSearchOfferingList.aspx?srch=CPLEX)
3. Install the CPLEX Optimization Studio.
4. Setup the CPLEX Python API [as described here](https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/set_up/Python_setup.html).

If you have problems installing CPLEX, check the [CPLEX user manual](http://www-01.ibm.com/support/knowledgecenter/SSSA5P/welcome) or the [CPLEX forums](https://www.ibm.com/developerworks/community/forums/html/forum?id=11111111-0000-0000-0000-000000002059).

## Development Timeline

***This package is in active development. Code may change substantially with each commit***

- simplify installation
- refactoring for future development
- open-source LP solver to create preprocessor
- scikit-learn API compatability
