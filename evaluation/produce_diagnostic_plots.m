function produce_diagnostic_plots(output)
figure(1);
colors=setup_plot(3);
plot(output.PR.rec, output.PR.prec, 'Color', colors(1,:), 'LineWidth', 2);
hold on;
plot(output.PR_nomisloc.rec, output.PR_nomisloc.prec, 'Color', colors(2,:), 'LineWidth', 2);
plot(output.PR_corrmisloc.rec, output.PR_corrmisloc.prec, 'Color', colors(3,:), 'LineWidth', 2);
xlabel('Recall', 'FontSize', 24);
ylabel('Precision', 'FontSize', 24);
legend({'Actual', 'No misloc', 'Corr misloc'}, 'FontSize', 24);
grid on;

%impact chart
figure(2);
clf
produce_impact_chart_all(output);



function colors = setup_plot(num)
colors = colormap(lines(num));

set(0,'DefaultAxesFontName', 'Times New Roman');
set(0,'DefaultAxesFontSize', 12);
set(0,'DefaultTextFontname', 'Times New Roman')
set(0,'DefaultTextFontSize', 12);


clf;
set(gcf, 'Color', 'white');
hold on;

