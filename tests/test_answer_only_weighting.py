import unittest,torch
from spatiallm_compressed_visual_objanc_full_attention.model.role_losses import role_mean,role_focal
class RoleLossTests(unittest.TestCase):
 def test_duplicate_anchor_invariant(self):
  a=role_mean(torch.tensor([2.,8.]),['OBJ','ANC'])[0]
  b=role_mean(torch.tensor([2.,8.,8.]),['OBJ','ANC','ANC'])[0]
  self.assertEqual(float(a),7.);self.assertEqual(float(a),float(b))
 def test_no_anchor_retains_obj_coefficient(self):
  self.assertEqual(float(role_mean(torch.tensor([2.]),['OBJ'])[0]),3.)
 def test_background_has_gradient_and_role_permutation(self):
  x=torch.zeros(1,3,4,requires_grad=True);p=torch.tensor([[[1,0,0,0],[0,1,0,0]]],dtype=torch.bool)
  valid=torch.ones(1,4,dtype=torch.bool);q=torch.tensor([0,1]);t=torch.tensor([0,1])
  total,details=role_focal(x,p,valid,[(q,t)],[['OBJ','ANC']],.25,2)
  total.backward();self.assertGreater(float(x.grad[0,2].abs().sum()),0);self.assertGreater(float(details['span_background']),0)
  y,d=role_focal(x.detach(),p[:,[1,0]],valid,[(q,torch.tensor([1,0]))],[['ANC','OBJ']],.25,2)
  self.assertAlmostEqual(float(total),float(y))

 def test_anchor_focal_group_duplication_invariant(self):
  v=torch.ones(1,4,dtype=torch.bool);p=torch.tensor([[[1,0,0,0],[0,1,0,0],[0,1,0,0]]],dtype=torch.bool)
  x=torch.zeros(1,3,4)
  _,a=role_focal(x,p[:,:2],v,[(torch.tensor([0,1]),torch.tensor([0,1]))],[['OBJ','ANC']],.25,2)
  _,b=role_focal(x,p,v,[(torch.tensor([0,1,2]),torch.tensor([0,1,2]))],[['OBJ','ANC','ANC']],.25,2)
  self.assertEqual(float(a['span_anc']),float(b['span_anc']))
