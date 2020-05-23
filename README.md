# QUD-dataset
Qatar university dataset (QUD) is an open access repository, which includes micro-moments power consumption footprints of different appliances. It is collected at Qatar university energy lab.  In the initial version of QUD, power usage footprints have been gathered for a period of more than 3 months until now. The collection campaign is still ongoing in order to cover a period of one year and other appliances.


The testbeds used to glean the data and the micro-momnets deployed to define power micro-momnets are described more thoroughly in the in the paper:  Y. Himeur, A. Alsalemi, F. Bensaali, A. Amira, Building power consumption datasets: Survey, taxonomy and future directions, Energy &amp; Buildings, 2020 (Submitted).  

QUD is an annotated dataset devoted for anomaly detection in power consumption. Five micro moment classes are defined, in which the first three ones represent normal consumption: “class 0: good usage”, “class 1: turn on”, and “class 2: turn off”. 
On the other hand, “class 4: excessive power consumption” and “class 5: consumption when outside” describe anomalous consumption.

Those wishing to use the dataset in academic work should cite this paper as the reference.  QUD_app-1.csv: this file includes the different kinds of data collected during the measurement campaign: Column 1: Date Column 2: Time Column 3: appID Column 4: occupancy pattern Column 5: Power consumption Column 6: Normalized power Column 7: Quantified power Column 8: Micro-moment class
