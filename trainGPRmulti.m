
function mdl = trainGPRmulti(X,Y,varargin)
% X : n×d predictors,  Y : n×P response matrix
% varargin : extra name–value pairs forwarded to fitrgp
P   = size(Y,2);
mdl = cell(P,1);
for p = 1:P
    mdl{p} = fitrgp(X, Y(:,p), varargin{:});
end
end
