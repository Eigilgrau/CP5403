import unittest
import pandas as pd
import phm_training_evidence as e

class TrainingEvidenceChecks(unittest.TestCase):
    def test_ties_and_missing_values(self):
        self.assertEqual(e.percentile_interval([7,7,7],7),[0.,1.])
        self.assertIsNone(e.percentile_interval([float('nan')],3))
        self.assertEqual(e.band([1,2,3,float('nan')])['n'],3)

    def test_train_only_and_no_self_case(self):
        t=pd.DataFrame({'case_id':['a'],'site_id':[1],'reference_label':['NoRecordedTarget']})
        e.assert_train_only(t,{1:'train'},['b'])
        with self.assertRaises(ValueError):e.assert_train_only(t,{1:'validation'},['b'])
        with self.assertRaises(ValueError):e.assert_train_only(t,{1:'train'},['a'])

    def test_constant_feature_has_overlap_and_zero_separation(self):
        t=pd.DataFrame({'reference_label':list(e.LABELS)})
        x=pd.DataFrame({'ch1_constant':[7.,7.,7.]})
        catalog,selected=e.build_catalog(t,x,['ch1_constant'])
        self.assertEqual(catalog['ch1_constant']['median_separation_over_pooled_central_span'],0)
        self.assertTrue(all(x['central_80_bands_overlap'] for x in catalog['ch1_constant']['pairwise_overlap']))

    def test_class_quantiles_and_reproducibility(self):
        t=pd.DataFrame({'reference_label':[c for c in e.LABELS for _ in range(3)]})
        x=pd.DataFrame({'ch2_feature':list(range(9))})
        a,selected=e.build_catalog(t,x,list(x.columns))
        self.assertEqual(a['ch2_feature']['by_class']['RuntimeExcessive']['median'],4)
        b,again=e.build_catalog(t,x,list(x.columns))
        self.assertEqual(a,b);self.assertEqual(selected,again)

    def test_retrieved_labels_verified_against_training(self):
        t=pd.DataFrame({'case_id':['a','b','c'],'site_id':[1,2,3],'reference_label':list(e.LABELS)}).set_index('case_id')
        r={'historical_cases':[{'case_id':i,'recorded_outcome':c} for i,c in zip(['a','b','c'],e.LABELS)]}
        s=e.retrieval_summary(r,t)
        self.assertTrue(s['tie_for_largest']);self.assertEqual(s['class_counts'],[1,1,1])
        r['historical_cases'][0]['recorded_outcome']='RuntimeExcessive'
        with self.assertRaises(ValueError):e.retrieval_summary(r,t)

if __name__=='__main__':unittest.main()
