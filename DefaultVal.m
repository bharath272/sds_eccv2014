% Function written by Dr. Adrian Ion
%
% This code is part of the extended implementation of the paper:
% 
% J. Carreira, C. Sminchisescu, Constrained Parametric Min-Cuts for Automatic Object Segmentation, IEEE CVPR 2010
% 

function DefaultVal(varargin)
	% assign default values to variables if they do not exist or are empty
	%
	% use string pairs 'var', 'value'
	% preceede the variable name with a '*' to set also it empty
	% 
	%
	% e.g. DefaultVal('p1', '1', '*p2', '2');
	%
	% will create p1 = 1 if it does not exist in the current scope
	% will create or set p2 = 2 if it does not exist, or it is empty

	assert(mod(nargin,2) == 0);
	
	for i=1:2:nargin
		if ~ischar(varargin{i}) || length(varargin{i})<1 || ~ischar(varargin{i+1}) length(varargin{i+1})<1
			error('Bad parameters passed to function. See function help.');
		end
		
		if (varargin{i}(1)=='*')
			cmd = sprintf('if ~exist(''%s'',''var'') || isempty(%s) %s = %s; end', varargin{i}(2:end), varargin{i}(2:end), varargin{i}(2:end), varargin{i+1});
		else
			cmd = sprintf('if ~exist(''%s'',''var'') %s = %s; end', varargin{i}, varargin{i}, varargin{i+1});
		end
		evalin('caller', cmd, 'error(''Requires and even number of (string names of possibly missing) variables & values'')') ;
	end
end