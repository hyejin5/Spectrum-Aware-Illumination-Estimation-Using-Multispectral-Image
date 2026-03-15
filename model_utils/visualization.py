import string
from model_utils.option import args
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import pdb
import shutil
import matplotlib as mat

import matplotlib.ticker as mticker
import colour

def setPlotStyle():    
    # plt.figure(dpi=300)  
    mat.rcParams['font.size'] = 15
    mat.rcParams['legend.fontsize'] = 12
    mat.rcParams['lines.linewidth'] = 2
    mat.rcParams['lines.color'] = 'r'
    # mat.rcParams['axes.grid'] = 1     
    mat.rcParams['axes.xmargin'] = 0.1     
    mat.rcParams['axes.ymargin'] = 0.1     
    mat.rcParams["mathtext.fontset"] = "dejavuserif" #"cm", "stix", etc.
    mat.rcParams['figure.dpi'] = 500
    mat.rcParams['savefig.dpi'] = 500




setPlotStyle()



def illumination_save(gt_L, output_norm, im_name, save_path):
    '''illum_graph'''
    setPlotStyle()

    x=np.linspace(380,730,36)

    gt_illum = gt_L.detach()
    gt_L_trans=gt_illum.cpu().numpy().transpose()
    # plt.plot(x, gt_L_trans, label='L_gt',linestyle=':' )
    # plt.savefig(save_path+'gt_illumination_%s'%(im_name))
    # plt.close()
    plt.xlabel('Wavelength', color = 'black')
    plt.ylabel('Normalized SPD', color = 'black')
    output_illum = output_norm[0].detach().cpu().numpy().transpose()
    plt.plot(x, gt_L_trans,linestyle='-' ,label='GT Illumination Spectrum', color='black' )
    plt.plot(x, output_illum, label='Output Illumination Spectrum', linestyle='dashed',color='mediumblue')
    plt.legend(loc='best')    # ncol = 2

    plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%i nm'))
    # ax2.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%i nm'))
    # plt.xticks(np.arange(380,790,10), rotation=45)
    # plt.yticks(np.arange(0,1,0.2))
    plt.tight_layout()
    plt.show()
    plt.savefig(save_path+'GT_illumination_%s'%(im_name),bbox_inches="tight")
    plt.close() 
    

def illumination_save_15CH(gt_L, output_norm, im_name, save_path):
    '''illum_graph'''
    x=np.linspace(380,830,15)

    gt_illum = gt_L[0].detach()
    gt_L_trans=gt_illum.cpu().numpy().transpose()
    # plt.plot(x, gt_L_trans, label='L_gt',linestyle=':' )
    # plt.savefig(save_path+'gt_illumination_%s'%(im_name))
    # plt.close()

    output_illum = output_norm[0].detach().cpu().numpy().transpose()
    plt.plot(x, gt_L_trans, label='L_gt',linestyle=':' )
    plt.plot(x, output_illum, label='L_output', linestyle='--')
    plt.legend(loc='best', ncol=2)    # ncol = 2
    plt.show()
    plt.savefig(save_path+'illumination_%s'%(im_name))
    plt.close()   

def gt_illumination_save_36CH(gt_L, im_name, save_path):
    '''illum_graph'''
    x_36=np.linspace(380,730,36)
    # pdb.set_trace()
    gt_illum = gt_L.detach().cpu().numpy()
    # plt.plot(x, gt_L_trans, label='L_gt',linestyle=':' )
    # plt.savefig(save_path+'gt_illumination_%s'%(im_name))
    # plt.close()

    plt.title('36CH GT illumination measured by spectrometer')

    plt.plot(x_36, gt_illum, label='L_gt',linestyle='-' )
    plt.legend(loc='best', ncol=2)    # ncol = 2
    plt.show()
    plt.savefig(save_path+'36CH_illumination_%s'%(im_name))
    plt.close()

def comparision_gt_illumination_save(gt_L_15,gt_L36, im_name, save_path):
    '''illum_graph'''
    x=np.linspace(380,735,13)
    x_36=np.linspace(380,730,36)

    gt_illum_15 = gt_L_15.detach().cpu().numpy()
    gt_illum_36 = gt_L36.detach().cpu().numpy()
    # plt.plot(x, gt_L_trans, label='L_gt',linestyle=':' )
    # plt.savefig(save_path+'gt_illumination_%s'%(im_name))
    # plt.close()
    # output_illum = output_norm[0].detach().cpu().numpy().transpose()
    plt.plot(x_36, gt_illum_36, label='L_gt_from_spectrometer(36CH)',linestyle='--' )

    plt.plot(x, gt_illum_15, label='L_gt_from_image(15CH)',linestyle='-' )
    # plt.plot(x, output_illum, label='L_output', linestyle='--')
    plt.legend(loc='lower right',bbox_to_anchor=(1.0,1.0), ncol=2)    # ncol = 2
    plt.show()
    plt.savefig(save_path+'Comparison_GT_illum_%s'%(im_name))
    plt.close()    

def illumination_save_3CH(gt_L, output_norm, im_name, save_path):
    '''illum_graph'''

    x=np.linspace(0,2,3)

    gt_illum = gt_L.detach().cpu().numpy()
    # pdb.set_trace()
    # gt_L_trans=gt_illum.
    gt_rgb = colour.XYZ_to_sRGB(gt_illum)
    gt_rgb=gt_rgb.squeeze()
    val_range = sum(gt_rgb)
    norm_gt_rgb = gt_rgb/val_range
    # plt.plot(x, gt_L_trans, label='L_gt',linestyle=':' )
    # plt.savefig(save_path+'gt_illumination_%s'%(im_name))
    # plt.close()

    output_illum = output_norm.detach().cpu().numpy()
    output_rgb = colour.XYZ_to_sRGB(output_illum)
    output_rgb=output_rgb.squeeze()
    output_val_range = sum(output_rgb)
    norm_output_rgb = output_rgb/output_val_range
    # pdb.set_trace()
    output_L_BGR = norm_output_rgb.copy()
    output_L_BGR[0]=norm_output_rgb[2]
    output_L_BGR[2]=norm_output_rgb[0]
    
    gt_norm_BGR = norm_gt_rgb.copy()
    gt_norm_BGR[0]=norm_gt_rgb[2]
    gt_norm_BGR[2]=norm_gt_rgb[0]


    x_val=['B', 'G', 'R']
    # gt_L_BGR=gt_L_BGR.squeeze()
    # output_norm_BGR=output_norm_BGR.squeeze()
    # least_critical_BGR=least_critical_BGR.squeeze()
    # plt.plot(x, gt_L_trans, label='L_gt',linestyle=':' )
    # plt.savefig(save_path+'gt_illumination_%s'%(im_name))
    # plt.close()
    # pdb.set_trace()
    # f, ax = plt.subplots(1,1)
    # ax.grid(True)

    # y_max = max(gt_L.max(),output_norm.max())
    plt.plot(x, gt_norm_BGR, label='GT RGB',linestyle='solid',marker='o',color='black')
    plt.plot(x, output_L_BGR, label='Output RGB', linestyle='dashed',marker='o', color='olivedrab')

    # ax.plot(x, least_critical_BGR, label='RGB without the least critical channel', linestyle='-', color='lightsteelblue', alpha=0.4)
    # ax.plot(x, least_critical_BGR, label='RGB without the least critical channel', linestyle='-', color='lightsteelblue', alpha=0.4)
    # plt.legend(loc='lower right', bbox_to_anchor=(1.0,1.0))    
    # plt.xlabel('Wavelength[nm]')
    plt.legend(loc='best')    # ncol = 2

    plt.ylabel('Normalized RGB Values')
    plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

    plt.xticks(x, x_val)


    plt.tight_layout()

    plt.show()
    plt.savefig(save_path+'RGB_illumination_%s' %im_name, bbox_inches="tight")
    plt.close()
    
    


def gt_rgb_save(image_rgb, im_name, rgb_save_path):
    rgb_sample = image_rgb.detach()
    rgb_sample = rgb_sample[0].cpu().numpy()
    rgb_sample = rgb_sample.transpose((1,2,0))

    file_name_rgb=rgb_save_path+'gt_rgb_%s.png' %(im_name)
    plt.imsave(file_name_rgb, rgb_sample)

def output_rgb_save(image_rgb, im_name, rgb_save_path): #output_image_rgb: W H 31
    file_name_rgb=rgb_save_path+'output_rgb_%s.png' %(im_name)
    image_rgb = image_rgb.clip(0,image_rgb.max())
    image_rgb = image_rgb/image_rgb.max()
    plt.imsave(file_name_rgb, image_rgb)