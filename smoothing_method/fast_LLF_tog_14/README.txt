% This is an implementation of the algorithm of edge-aware detail
% manipulation described in the paper:
%
% Fast Local Laplacian Filters: Theory and Applications. 
% Mathieu Aubry, Sylvain Paris, Samuel W. Hasinoff, Jan Kautz, and Fredo Durand. 
% ACM Transactions on Graphics 2014
%
%
% The key scripts and functions are:
%   llf.m                  - the algorithm of the standard version of the filter described in the paper
%   llf_general.m          - the algorithm of the general version of the filter
%   style_transfer.m       - perform the style transfer algorithm with two grey-scale images
%   demo.m                 - some examples of the LLF
%   demo_style_transfer.m  - example of style transfer 
%
% Includes Laplacian pyramid routines adapted from Tom Mertens'
% Matlab implementation, modified by Samuel W. Hasinoff.
%
% mathieu.aubry@m4x.org, March 2014
%