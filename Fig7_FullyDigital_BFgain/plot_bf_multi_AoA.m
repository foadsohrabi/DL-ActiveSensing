clear; close all
% figure;
figure('Renderer', 'painters', 'Position', [360 150 620 485]);
set(0,'defaulttextInterpreter','latex');
color1=[0, 0.4470, 0.7410];
color2=[0.8500, 0.3250, 0.0980];
color3=[0, 0.5, 0];
color4=[1, 0, 0];
color5=[0.4940, 0.1840, 0.5560];
color6=[0.3010, 0.7450, 0.9930];
fs2 = 14;

pilots_omp = [1,3,7,11,15,19,23];
bf_gain_omp = [1.0607432077388828,2.71133237,7.59000374041721,14.9929870530275,...
    22.25353798987,28.962189419,34.91724168507398];
bf_gain_omp = 10*log10(bf_gain_omp);
bf_gain_opt = 18.0547570*ones(1,length(pilots_omp));


pilots_DNN = [1,3,7,11,15,19,23];
bf_gain_DNN = [2.8575122,6.491461,11.054041,13.447309,14.816213,15.553158,16.080421];

pilots_RNN = [1,3,7,11,15,19,23];
bf_gain_RNN = [5.5086823,11.004192,14.264064,15.469575,16.270512,16.683128,17.206976];

pilots_DNN_trainable = [1,3,7,11,15,19,23];
bf_gain_DNN_trainable = [5.635976,10.175379,12.969524,14.429845,15.459808,16.150192,16.570518];
   
    
plot(pilots_omp,bf_gain_opt,'o-','color',color1,'lineWidth',2,'markersize',8);
hold on 
plot(pilots_omp,bf_gain_omp,'v-','color',color2,'lineWidth',2,'markersize',8);
hold on
plot(pilots_DNN,bf_gain_DNN,'s-','color',color3,'lineWidth',2,'markersize',8);
hold on
plot(pilots_DNN_trainable,bf_gain_DNN_trainable,'+-','color',color4,'lineWidth',2,'markersize',8);
hold on
plot(pilots_RNN,bf_gain_RNN,'-*','color',color5,'lineWidth',2,'markersize',8);
grid on
lg=legend('MRT w$/$ perfect CSI', 'MRT w/ OMP channel estimation (random sensing, fixed)',...
    'DNN-based design (random sensing vectors, fixed)',...
    'DNN-based design (learned sensing vectors, fixed)',...
    'Proposed active sensing method','Location','southeast');
set(lg,'Fontsize',fs2-3.2);
set(lg,'Interpreter','latex');
xticks([1,3:4:23]);
xlabel('Sensing Time Frame $T$','Interpreter','latex','FontSize',fs2);
ylabel('Average Beamforming Gain (dB)','Interpreter','latex','FontSize',fs2);
