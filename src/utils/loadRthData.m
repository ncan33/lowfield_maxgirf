% ***************************************************************************
% Copyright 2015 HeartVista Inc.  All rights reserved.
% Contact: HeartVista, Inc. <rthawk-info@heartvista.com>
%
% This file is part of the RTHawk system.
%
% $HEARTVISTA_BEGIN_LICENSE$
%
% THIS IS UNPUBLISHED PROPRIETARY SOURCE CODE OF HEARTVISTA
% The copyright notice above does not evidence any
% actual or intended publication of such source code.
%
% $HEARTVISTA_END_LICENSE$
%
% ***************************************************************************
%
%
% loadRthData.m
% Read data and header from the file exported by RTHawk
% This can take several minutes if the files are big
%
%     input: fileName -- the name of the file to which data have
%                        been exported using RthReconImageExport.cpp
%     output: data -- the image data
%             header -- the image data header
%             kspace -- the acquisition trajectory (kx ky density
%                       for each sample)
%

function [data, header, kspace] = loadRthData(fileName)

  % Open file
  fip = fopen(fileName, 'r', 'l');
  if (fip == -1)
      tt = sprintf('File %s not found\n', fileName);
      error(tt);
      return;
  end

  % Check that this is an RTHawk file
  [magic, count] = fread(fip, 4, 'char');
  if (count ~= 4)
      tt = sprintf('Cannot read file format\n');
      error(tt);
      return;
  end

  if (sum(magic' ~= ['H' 'O' 'C' 'T']))
      tt = sprintf('Invalid file format\n');
      error(tt);
      return;
  end

  % Read version, It should be 1, 2, or 3
  [version, count] = fread(fip, 1, 'int');
  if (count ~= 1)
      tt = sprintf('Cannot read version\n');
      error(tt);
      return;
  end

  if version > 3
      header = [];
  else
      header = struct();
  end

  i=1;
  data = [];
  kspace = [];
  while(1)
      % Read number of (key, value) pairs contained in the header
      [hashcount, count] = fread(fip, 1, 'int');
      if (count ~= 1)
          %fprintf(1,'Successfully read %d frames\n',i-1);
          break;
      end

      if version > 3
          headerData = char(fread(fip, hashcount, '*char'));
          header = [header, jsondecode(convertCharsToStrings(headerData))];
	 else
          % Read header
          for k=1:hashcount
              stringLength = fread(fip, 1, 'int');
              key = fread(fip, stringLength, '*char')';
              value = fread(fip, 1, 'double');
              header(i).(key) = value;
          end
	 end

      if (version >= 2)
          % Read kspace size
          [samples, count] = fread(fip, 1, 'int');
          if (count ~=1)
              tt = sprintf('Cannot read kspace size\n')
              error(tt);
              return;
          end

          % Read kspace
          kspace(:,:,i) = fread(fip, [1, samples], 'float');
      end


      if(isfield(header(i), 'dataSize'))
          dataSize = round(header(i).dataSize);
          % Read next block of data
          [tmpData, count] = fread(fip, [2,dataSize], 'float');
          if(count ~= 2*dataSize)
              fprintf(1,'End of file reached. Expected %d blocks of data but only read %d blocks.\n',dataSize,count);
              fprintf(1,'Successfully read %d frames\n',i-1);
              break;
          end
          data(:,:,i) = tmpData;
      else
          % Read all remaining data
          [data, count] = fread(fip, [2,inf], 'float');

          % Print header
          header

          break;
      end

      i = i+1;
  end
end
