

def a():
    self.mergeOn = mergeOn
    self.dumOn = dumOn

    oversampler=SMOTEENN()
    testKwargs={'test_size': .3}

    excludeColsRisk = list(excludeColsRisk)
    subDfRisk = dfRisk[dumColsScores]
    dfRisk.drop(dumColsScores, inplace=True, axis=1)
    dumScore = pd.get_dummies(subDfRisk,  columns=[i for i in dumColsScores
                                                   if i != 'n_scores'])

    risk = pd.DataFrame(data=dfRisk.company_id, index=dfRisk.index)
    for i in dumScore.drop(['n_scores'], axis=1):
        risk[i] = dumScore[i] * dumScore.n_scores

    self.g = risk.groupby('company_id')
    groupedRisk = risk.groupby('company_id').sum()

    dumRiskAttributes = pd.get_dummies(dfRisk[[i for i in dfRisk if i not in dumColsScores]], columns=dumCols, drop_first=True)
    dfRiskMerge = dumRiskAttributes.merge(groupedRisk, left_on='company_id', right_index=True)

    df = dfCompanies.merge(dfRiskMerge, how='left',
                           left_on='company_id', right_index=True)\
                    .fillna(0)

    df.bank_account_type = (df.bank_account_type == 'Checking')

    df = df.drop(['company_id_x', 'company_id_y'], axis=1).astype(int)

    train, test = tts(df, stratify=df.is_fraud, **testKwargs)

    designCols = [i for i in train if i != target]
    self.designTrain = train[designCols]
    self.designTest = test[designCols]

    self.targetTrain = train[target]
    self.targetTest = test[target]

    designTrain, targetTrain =\
        oversampler.fit_sample(self.designTrain, self.targetTrain)
    self.designTrain = pd.DataFrame(designTrain, columns=self.designTrain.columns)
    self.targetTrain = pd.Series(targetTrain, name=self.targetTrain.name)
    # dropCols = pd.concat([self.designTrain, pd.DataFrame(self.targetTrain)], axis=1).corr()[target].isnull().reset_index()
    # self.designTrain.drop(dropCols['index'].ix[dropCols.is_fraud], inplace=True, axis=1) # Drop columns that will produce singular design 
    # self.designTest.drop(dropCols['index'].ix[dropCols.is_fraud], inplace=True, axis=1)



    # def fit(self, risks, companies, target, riskCol, keyCols):
        # self._build_risk_weights(risks, riskCol, keyCols)


    # def _build_risk_weights(self, risks, riskCol, keyCols):
        # positive = risks.ix[risks[riskCol] == 1]
        # self.scores = (positive.groupby(keyCols).count().astype(float)
                       # ['n_scores'] /
                       # risks.groupby(keyCols).count().astype(float)
                       # ['n_scores']).fillna(0)

    # def transform(self, X):

        # XScores = X.merge(self.scores.reset_index(), how='left',
                          # on=['signal_group', 'score_value', 'metric_type'])

        # groupedSum = XScores.groupby('company_id').sum()
        # groupedSum['risk_sum'] = groupedSum.n_scores_x * groupedSum.n_scores_y
        # groupedSum['risk_weight'] = groupedSum.risk_sum / groupedSum.n_scores_x
