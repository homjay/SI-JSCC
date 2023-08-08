% verify jscc result
%========================================================
% input signal mat
eval_snr=0;
%========================================================
signal_mat='./z.mat';
load(signal_mat);
%========================================================
% convert real z to complex z
z_complex=complex(z(1,:), z(2,:));
%========================================================
% add awgn noise
z_noised = awgn(z_complex, eval_snr, 'measured');
%========================================================
% save sig
save("./z_noised.mat", "z_noised");

