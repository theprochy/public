function [front_door, rear_door, front_hinges, rear_hinges, front_sensors, rear_sensors] = parse(filename, doors)
    front_door = [];
    rear_door = [];
    front_hinges = zeros(2, 3);
    rear_hinges = zeros(2, 3);
    front_sensors = zeros(2, 3);
    rear_sensors = zeros(2, 3);
    fid = fopen(filename);
    line = fgetl(fid);
    while ischar(line)
        exp = "#\d+=CARTESIAN_POINT";
        if regexp(line, exp)
            line = line(regexp(line,"=")+17:regexp(line, "))"));
            line = split(erase(erase(erase(line, "("), ")"), "'"), ",");
            desc = line{1};
            x = str2double(line{2});
            y = str2double(line{3});
            z = str2double(line{4});
            if startsWith(desc, 'door')
                idx = str2double(desc(regexp(desc, "\d+"):end));
                if startsWith(desc, 'door FD Left')
                    front_door(idx, :) = [x y z];
                elseif startsWith(desc, 'door RD Left') && numel(doors) == 4
                    rear_door(idx, :) = [x y z];
                end
            elseif startsWith(desc, 'hinge')
                if startsWith(desc, 'hinge FD Left')
                    if regexp(desc, 'bottom')
                        front_hinges(1, :) = [x y z];
                    elseif regexp(desc, 'top')
                        front_hinges(2, :) = [x y z];
                    end
                elseif startsWith(desc, 'hinge RD Left') && numel(doors) == 4
                    if regexp(desc, 'bottom')
                        rear_hinges(1, :) = [x y z];
                    elseif regexp(desc, 'top')
                        rear_hinges(2, :) = [x y z];
                    end
                end
            elseif startsWith(desc, 'sensor')
                if startsWith(desc, 'sensor FD1 Left')
                    front_sensors(1, :) = [x y z];
                elseif startsWith(desc, 'sensor FD2 Left')
                    front_sensors(2, :) = [x y z];
                elseif startsWith(desc, 'sensor RD1 Left') && numel(doors) == 4
                    rear_sensors(1, :) = [x y z];
                elseif startsWith(desc, 'sensor RD2 Left') && numel(doors) == 4
                    rear_sensors(2, :) = [x y z];
                end
            end
        end
        line = fgetl(fid);
    end
    fclose(fid);
end