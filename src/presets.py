# select some repsentative data 
# following the presets we are going to use
preset_dict = {
    'mul_vegetation': {'width':325,'channel_count':8,'channels':[7,5,3],'subfolder':'MUL'},
    'mul_urban': {'width':325,'channel_count':8,'channels':[8,7,5],'subfolder':'MUL'},     
    'mul_blackwater': {'width':325,'channel_count':8,'channels':[7,8,1],'subfolder':'MUL'},     
    'mul_ir1': {'width':325,'channel_count':8,'channels':[7,7,7],'subfolder':'MUL'}, 
    'mul_ir2': {'width':325,'channel_count':8,'channels':[8,8,8],'subfolder':'MUL'},      
    'mul_naive1': {'width':325,'channel_count':8,'channels':[1,2,3],'subfolder':'MUL'},
    'mul_naive2': {'width':325,'channel_count':8,'channels':[4,5,6],'subfolder':'MUL'},
    'mul_naive3': {'width':325,'channel_count':8,'channels':[6,7,8],'subfolder':'MUL'},
    
    'pan': {'width':1300,'channel_count':1,'channels':[1,1,1],'subfolder':'PAN'},
    
    'rgb_ps': {'width':1300,'channel_count':3,'channels':[1,2,3],'subfolder':'RGB-PanSharpen'},
    
    'mul_ps_vegetation': {'width':1300,'channel_count':8,'channels':[7,5,3],'subfolder':'MUL-PanSharpen'},
    'mul_ps_urban': {'width':1300,'channel_count':8,'channels':[8,7,5],'subfolder':'MUL-PanSharpen'},     
    'mul_ps_blackwater': {'width':1300,'channel_count':8,'channels':[7,8,1],'subfolder':'MUL-PanSharpen'},     
    'mul_ps_ir1': {'width':1300,'channel_count':8,'channels':[7,7,7],'subfolder':'MUL-PanSharpen'}, 
    'mul_ps_ir2': {'width':1300,'channel_count':8,'channels':[8,8,8],'subfolder':'MUL-PanSharpen'},
    'mul_ps_naive1': {'width':1300,'channel_count':8,'channels':[1,2,3],'subfolder':'MUL-PanSharpen'}, 
    'mul_ps_naive2': {'width':1300,'channel_count':8,'channels':[4,5,6],'subfolder':'MUL-PanSharpen'},    
    'mul_ps_naive3': {'width':1300,'channel_count':8,'channels':[6,7,8],'subfolder':'MUL-PanSharpen'},  
    
    'mul_ps_8channel': {'width':1300,'channel_count':8,'channels':[1,2,3,4,5,6,7,8],'subfolder':'MUL-PanSharpen'},
    'mul_8channel': {'width':325,'channel_count':8,'channels':[1,2,3,4,5,6,7,8],'subfolder':'MUL'},      
}
