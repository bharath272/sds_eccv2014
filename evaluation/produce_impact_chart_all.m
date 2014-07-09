function produce_impact_chart_all(outputs)
clf
set(gcf, 'Color', 'white');
AP=0; L=0; B=0; S=0; Lfix=0;
for i=1:numel(outputs)
    output=outputs(i);
    AP=AP+output.PR.ap*100./numel(outputs);
    L=L+output.PR_nomisloc.ap*100./numel(outputs);
    B=B+output.PR_nobg.ap*100./numel(outputs);
    S=S+output.PR_nosim.ap*100./numel(outputs);
    Lfix=Lfix+output.PR_corrmisloc.ap*100./numel(outputs);
end
hold on;
barh(3, max(Lfix-AP,0), 'FaceColor', [79 129 189]/255);
barh(3, max(L-AP,0), 'FaceColor', [79 129 189]/255*0.8);
barh(2, max(S-AP,0), 'FaceColor', [192 80 77]/255);
barh(1, max(B-AP,0), 'FaceColor', [128 100 162]/255);
ylim([0.5 3.5]);

xlim = [0 ceil((max([Lfix L S B]-AP)+0.005)*20)/20];
set(gca, 'xlim', xlim);
set(gca, 'xminortick', 'on');
set(gca, 'ticklength', get(gca, 'ticklength')*4);
set(gca, 'ytick', 1:3)
set(gca, 'yticklabel', {'B', 'S', 'L'}, 'FontSize', 20);
xlabel('Change in AP^r (percentage points)', 'FontSize', 24);

