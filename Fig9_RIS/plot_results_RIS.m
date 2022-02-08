clear;
close all
figure('Renderer', 'painters', 'Position', [360 150 620 485]);
set(0,'defaulttextInterpreter','latex');
color1=[0, 0.4470, 0.7410];
color2=[0.8500, 0.3250, 0.0980];
color3=[0, 0.5, 0];
color4=[1, 0, 0];
color5=[0.4940, 0.1840, 0.5560];
color6=[0.3010, 0.7450, 0.9930];
fs2=14;


Pilot_rnn = [1,2,3,4,5,7,9];
Pilot = [1,2,3,4,5,7,9];
optimal = 35.64989*ones(1,length(Pilot));
rnn = [25.29668,30.25198,32.959864,34.02751207,34.598159,34.9427,35.02643];
DNN_trainable_w = [25.294623,29.740290,32.04726,33.437685,34.27970,34.849183,35.00075];
DNN = [24.21416,27.079641,29.8928546,32.29015,33.59381914,34.56213,34.9161124];
lmmse = [20.38243213986474, 22.19929718949977, 23.890383052645458, 24.88120869775, 25.4454477164903, 26.781319429118795, 28.210068902062112];

plot(Pilot,optimal,'s-','color',color1,'lineWidth',2,'markersize',8);
hold on
plot(Pilot_rnn,rnn,'-*','color',color5,'lineWidth',2,'markersize',8);
hold on
plot(Pilot,DNN_trainable_w,'-o','color',color2,'lineWidth',2,'markersize',8);
grid on
plot(Pilot,DNN,'-<','color',color3,'lineWidth',2,'markersize',8);
grid on
plot(Pilot,lmmse,'-x','color',color4,'lineWidth',2,'markersize',8);

lg=legend('Phase matching  w$/$ perfect CSI',...
    'Proposed active sensing method',...
    'DNN-based design (random sensing vectors, fixed)',...
    'DNN-based design (learned sensing vectors, fixed)',...
    'Phase matching w$/$ LMMSE CSI est.',...
    'Location','southeast');
set(lg,'Fontsize',fs2-3);
set(lg,'Interpreter','latex');
xlabel('Sensing Time Frame $T$','Interpreter','latex','FontSize',fs2);
ylabel('Average Beamforming Gain (dB)','Interpreter','latex','FontSize',fs2);
