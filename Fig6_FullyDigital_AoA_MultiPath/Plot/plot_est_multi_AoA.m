clc;
close all;
clear all;

snr = -10:5:25;
color1=[0, 0.4470, 0.7410];
color2=[0.8500, 0.3250, 0.0980];
color3=[0, 0.5, 0];
color4=[1, 0, 0];
color5=[0.4940, 0.1840, 0.5560];
color6=[0.3010, 0.7450, 0.9930];
load('OMPtau4.mat');
mse_omp(1) = mse_aoa;
load('OMPtau4.mat');
mse_omp(2) = mse_aoa;
load('OMPtau8.mat');
mse_omp(3) = mse_aoa;
load('OMPtau12.mat');
mse_omp(4) = mse_aoa;
load('OMPtau16.mat');
mse_omp(5) = mse_aoa;
% load('OMPtau20.mat');
% mse_omp(6) = mse_aoa;

load('data_RNN_multiAoAs_tau2.mat');
mse_rnn_coherent(1) = performance(end);
load('data_RNN_multiAoAs_tau4.mat');
mse_rnn_coherent(2) = performance(end);
load('data_RNN_multiAoAs_tau8.mat');
mse_rnn_coherent(3) = performance(end);
load('data_RNN_multiAoAs_tau12.mat');
mse_rnn_coherent(4) = performance(end);
load('data_RNN_multiAoAs_tau16.mat');
mse_rnn_coherent(5) = performance(end);
% load('data_RNN_multiAoAs_tau20.mat');
% mse_rnn_coherent(5) = performance(end);

load('data_RNN_multiAoAs_nonCoh_tau2.mat');
mse_rnn_noncoherent(1) = performance(end);
load('data_RNN_multiAoAs_nonCoh_tau4.mat');
mse_rnn_noncoherent(2) = performance(end);
load('data_RNN_multiAoAs_nonCoh_tau8.mat');
mse_rnn_noncoherent(3) = performance(end);
load('data_RNN_multiAoAs_nonCoh_tau12.mat');
mse_rnn_noncoherent(4) = performance(end);
load('data_RNN_multiAoAs_nonCoh_tau16.mat');
mse_rnn_noncoherent(5) = 0.0177;%performance(end);


T = [2,4,8,12,16];
T2 = [2,4,8,16];

figure('Renderer', 'painters', 'Position', [360 150 620 485]);
set(0,'defaulttextInterpreter','latex');
semilogy(T,mse_omp,'-','color',color3,'linewidth',2,'markersize',6);
hold on
semilogy(T,mse_rnn_noncoherent,'--*','color',color5,'lineWidth',2,'markersize',8);
hold on
semilogy(T,mse_rnn_coherent,'-*','color',color5,'lineWidth',2,'markersize',8);

grid;
fs2 = 14;
h = xlabel('Sensing Time Frame $T$','FontSize',fs2);
get(h)
h = ylabel('Average MSE: $E\left[ \sum_{\ell=1}^{L_p}(\phi_\ell - \hat{\phi}_\ell)^2\right]$','FontSize',fs2);
get(h)
lg = legend({'OMP w$/$ random fixed beamforming','Proposed active sensing method (noncoherent)','Proposed active sensing method (coherent)'},'Interpreter','latex','Location','southeast');
set(lg,'Fontsize',fs2-3);
set(lg,'Location','southwest');
ylim([10^(-3)*5,1])
