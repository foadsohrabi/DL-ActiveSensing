clc;
close all;
clear all;
color1=[0, 0.4470, 0.7410];
color2=[0.8500, 0.3250, 0.0980];
color3=[0, 0.5, 0];
color4=[1, 0, 0];
color5=[0.4940, 0.1840, 0.5560];
color6=[0.3010, 0.7450, 0.9930];

snrdB = -10:5:25;

load('data_baselines_hiePM_hieBS_OMPrandom_hyb.mat');
perf_bisec = mse_bisec;
perf_OMP = mse_OMP;
perf_AL_perfect = mse_AL_perfect_OS;

% load('data_DNN_known_alpha_EST_hybrid_phase.mat');
% performance_Hyb_phase = performance;

load('data_DNN_known_alpha_EST_hybrid_norm.mat');
performance_Hyb_norm = performance;

load('data_DNN_unknown_alpha_Kalman_EST_hyb_norm.mat');
performance_Kalman_Hyb_norm = performance;

load('data_DNN_unknown_alpha_MMSE_updatePIs_EST_hyb_norm.mat');
performance_MMSE_updatePI_Hyb_norm = performance;

load('data_RNN_noncoherent_modulus.mat');
performance_RNN_noncoh = performance;

load('data_RNN_coherent_modulus.mat');
performance_RNN_coh = performance;

figure('Renderer', 'painters', 'Position', [360 150 620 485]);
set(0,'defaulttextInterpreter','latex');

semilogy(snrdB,perf_bisec,'-.k','linewidth',2,'markersize',6);
hold on;
semilogy(snrdB,perf_OMP,'-','color',color3,'linewidth',2,'markersize',6);
hold on;
% semilogy(snrdB,performance_OMP_DNN,'-*','color',color3,'linewidth',2,'markersize',8);
% hold on;
semilogy(snrdB,perf_AL_perfect,'-sr','linewidth',2,'markersize',8);
hold on;
% semilogy(snrdBvec,performance_MMSE_updatePI_Hyb_norm,'--ob','linewidth',2,'markersize',6);
hold on;
semilogy(snrdB,performance_Kalman_Hyb_norm,'-.>','color',color6,'linewidth',2,'markersize',7);
hold on;
semilogy(snrdB,performance_Hyb_norm,'->','color',color6,'lineWidth',2,'markersize',8);
hold on
semilogy(snrdB,performance_RNN_noncoh,'--*','color',color5,'lineWidth',2,'markersize',8);
hold on
semilogy(snrdB,performance_RNN_coh,'-*','color',color5,'lineWidth',2,'markersize',8);

% semilogy(snrdBvec,performance_Hyb_phase,'-->g','linewidth',2,'markersize',6);
% hold on;

grid;
fs2 = 14;
h = xlabel('SNR (dB)','FontSize',fs2);
get(h)
h = ylabel('Average MSE: $E [ (\phi - \hat{\phi})^2 ]$','FontSize',fs2);
get(h)
lg = legend({'hieBS (noncoherent)',... 
             'OMP w$/$ random fixed beamforming (coherent)',...
             'hiePM w$/$ known $\alpha$ (coherent)',...
             'DNN w$/$ Kalman tracking for $\alpha$ (coherent)',...
             'DNN w$/$ known $\alpha$',...
             'Proposed active sensing method (noncoherent)',...
             'Proposed active sensing method (coherent)'},'Interpreter','latex','Location','southeast');
set(lg,'Fontsize',fs2-3);
set(lg,'Location','southwest');
ylim([10^-5,1]);
xlim([-10,25])
% xticks(-10:2.5:25)

