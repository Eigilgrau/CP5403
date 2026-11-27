import unittest
import pandas as pd
import numpy as np
from phm_temporal_retrieval_revision import recent

class TemporalChecks(unittest.TestCase):
    def window(self):
        t=pd.Timestamp('2020-01-01 12:00')
        rows=[]
        for comp,values in [('a',[1,2,3,4]),('b',[10,8,6,4])]:
            for minute,value in zip([45,30,15,0],values):
                rows.append({'component':comp,'bin_end':t-pd.Timedelta(minutes=minute),'ch1':value,'ch2':value,'ch5':value})
        return t,pd.DataFrame(rows)

    def test_preserves_opposing_component_changes(self):
        t,w=self.window();r=recent(w,t,2)
        self.assertEqual(r['ch2_last60_change_min'],-6)
        self.assertEqual(r['ch2_last60_change_max'],3)
        self.assertEqual(r['ch2_last60_coverage_fraction'],1)
        self.assertEqual(r['ch2_last15_level_median'],4)

    def test_sparse_data_not_treated_as_stability(self):
        t,w=self.window();w=w[w.bin_end==t]
        r=recent(w,t,2)
        self.assertTrue(np.isnan(r['ch2_last60_change_median']))
        self.assertEqual(r['ch2_last60_coverage_fraction'],0)
        self.assertEqual(r['ch2_last15_coverage_fraction'],1)

    def test_future_measurement_rejected(self):
        t,w=self.window();w.loc[0,'bin_end']=t+pd.Timedelta(minutes=15)
        with self.assertRaises(ValueError):recent(w,t,2)

    def test_duplicate_bins_rejected(self):
        t,w=self.window();w=pd.concat([w,w.iloc[:1]],ignore_index=True)
        with self.assertRaises(ValueError):recent(w,t,2)

if __name__=='__main__':unittest.main()
