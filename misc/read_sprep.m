function [sp, reg2sp] = read_sprep(spfile, reg2spfile)
reg2sp=imread(reg2spfile);
B=textread(spfile, '%d');

sz=B(1:2);
sp=reshape(B(3:end),sz(:)');
