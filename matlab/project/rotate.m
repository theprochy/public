function [W, N] = rotate(v)
    if norm(v)
        u = v / norm(v);
    else
        u = v;
    end
    W = [0 -u(3) u(2); u(3) 0 -u(1); -u(2) u(1) 0];
    N = @(phi) eye(3) + sin(phi) * W + 2 * sin(phi / 2)^2 * W^2;
    