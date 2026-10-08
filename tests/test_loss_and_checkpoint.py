from types import SimpleNamespace
import torch
from torch import nn
from spatiallm_compressed_visual_objanc_full_attention.model.segmentation_model import ObjAncSegmentationModel
from spatiallm_compressed_visual_objanc_full_attention.model.segmentation_decoder import ObjAncFullCycleDecoder

def test_auxiliary_sum_and_three_loss_weights():
    stage=ObjAncSegmentationModel.__new__(ObjAncSegmentationModel); nn.Module.__init__(stage)
    stage.loss_cfg={'aux_weight':.5}; stage.ranking_cfg={'weight':.5}; stage.answer_lm_weight=.1
    aggregate=getattr(stage,'_total_loss',None)
    assert callable(aggregate), 'missing testable objective aggregation'
    primary=torch.tensor(2.,requires_grad=True)
    auxiliary=[torch.tensor(1.,requires_grad=True) for _ in range(9)]
    total,aux=aggregate(primary,auxiliary,torch.tensor(3.),torch.tensor(4.))
    assert aux.item()==9
    assert abs(total.item()-8.4)<1e-5
    total.backward()
    assert all(t.grad.item()==.5 for t in auxiliary)

def test_decoder_896_text_and_gradients():
    torch.manual_seed(1)
    decoder=ObjAncFullCycleDecoder(visual_dim=8,text_dim=896,hidden_dim=16,
             num_queries=50,num_layers=9,num_heads=4,feedforward_dim=32)
    visual=torch.randn(5,8,requires_grad=True); text=torch.randn(4,896,requires_grad=True)
    result=decoder([visual],[text],[torch.randn(5,3)],[torch.tensor([0,1,1,3])])
    assert result['pred_masks'].shape==(1,50,5)
    assert len(result['aux_outputs'])==9
    (result['pred_masks'].square().mean()+result['pred_logits'].square().mean()).backward()
    assert text.grad is not None and text.grad.abs().sum()>0
    assert visual.grad is not None and visual.grad.abs().sum()>0

def test_layer_loss_supervises_only_matched_masks_and_rematches():
    stage=ObjAncSegmentationModel.__new__(ObjAncSegmentationModel); nn.Module.__init__(stage)
    stage.loss_cfg={'span_weight':6,'mask_bce_weight':5,'dice_weight':2,'focal_alpha':.25,'focal_gamma':2}
    stage.matcher_cfg={'span_weight':2,'mask_bce_weight':15,'dice_weight':6}
    valid=torch.tensor([[True]]); positive=torch.tensor([[[True,False]]])
    masks=torch.tensor([[[5.,5.,-5.],[-5.,-5.,5.]]],requires_grad=True)
    output={'pred_masks':masks,'pred_logits':torch.zeros(1,2,2,requires_grad=True),'text_valid_mask':torch.ones(1,2,dtype=torch.bool)}
    targets=[torch.tensor([[1.,1.,0.]])]
    loss,*other=stage._layer_loss(output,targets,positive,valid,[["OBJ"]])
    assert other[3][0][0].item()==0
    loss.backward(); assert masks.grad[0,1].abs().sum()==0
    flipped={**output,'pred_masks':masks.detach().flip(1)}
    assert stage._layer_loss(flipped,targets,positive,valid,[["OBJ"]])[4][0][0].item()==1

def test_conditional_trainable_parameters_keep_zero_gradient_graph():
    stage=ObjAncSegmentationModel.__new__(ObjAncSegmentationModel); nn.Module.__init__(stage)
    stage.used=nn.Parameter(torch.ones(2)); stage.unused=nn.Parameter(torch.ones(3))
    stage.frozen=nn.Parameter(torch.ones(()),requires_grad=False)
    loss=stage.used.square().sum()
    connected=stage._connect_trainable_graph(loss)
    assert connected.item()==loss.item()
    connected.backward()
    assert torch.equal(stage.used.grad,torch.full((2,),2.))
    assert torch.equal(stage.unused.grad,torch.zeros(3))
    assert stage.frozen.grad is None

def test_final_and_aux_layer_group_coefficients_are_identical():
 stage=ObjAncSegmentationModel.__new__(ObjAncSegmentationModel);nn.Module.__init__(stage)
 stage.loss_cfg={'span_weight':6,'mask_bce_weight':5,'dice_weight':2,'focal_alpha':.25,'focal_gamma':2}
 stage.matcher_cfg={'span_weight':2,'mask_bce_weight':15,'dice_weight':6}
 out={'pred_masks':torch.zeros(1,3,2),'pred_logits':torch.zeros(1,3,4),'text_valid_mask':torch.ones(1,4,dtype=torch.bool)}
 positive=torch.tensor([[[1,0,0,0],[0,1,0,0]]],dtype=torch.bool);valid=torch.ones(1,2,dtype=torch.bool)
 target=[torch.tensor([[1.,0.],[0.,1.]])]
 final=stage._layer_loss(out,target,positive,valid,[['OBJ','ANC']]);details=dict(stage._last_layer_details)
 aux=stage._layer_loss(out,target,positive,valid,[['OBJ','ANC']])
 for a,b in zip(final[:4],aux[:4]):torch.testing.assert_close(a,b)
 assert float(details['bce_obj_weighted'])==1.5*float(details['bce_obj'])
 assert float(details['bce_anc_weighted'])==.5*float(details['bce_anc'])
 assert float(details['span_background'])>0
