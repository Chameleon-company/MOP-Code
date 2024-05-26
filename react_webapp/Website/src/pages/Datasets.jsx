import React from 'react';
import Header from '../components/Header';
import Footer from '../components/Footer';
import '../styles/Datasets.css';

const Datasets = () => {
  return (
    <div>
      <Header />
      <main>
        <div>
          <p className='title'>What are you looking for today?</p>
          <table>
            <tr>
              <td><img src="src/assets/people.png" alt="People" width="220px" /></td>
              <td></td>
              <td><img src="src/assets/cloud.png" alt="Cloud" width="220px" /></td>
            </tr>
            <tr>
              <td>
                <div class="container">
                  <div class="table_top hover:bg-green-200">People</div>
                  <div class="table_bottom hover:bg-green-200">Property</div>
                </div>
              </td>
              <td><img src="src/assets/recycle.png" alt="Recycle" width="220px" /></td>
              <td> <div class="container">
                <div class="table_top hover:bg-green-200">Sensors</div>
                <div class="table_bottom hover:bg-green-200">Transportation</div>
              </div></td>
            </tr>
            <tr>
              <td><img src="src/assets/house.png" alt="House" width="220px" /></td>
              <td><div class="container">
                <div class="table_top hover:bg-green-200">Environments</div>
              </div></td>
              <td><img src="src/assets/tram.png" alt="Tram" width="220px" /></td>
            </tr>
          </table>
        </div>
      </main>
      <Footer />
    </div>
  );
};

export const datasetsSearchableContents = [
  "People",
  "Property",
  "Sensors",
  "Transportation",
  "Environments",
];

export default Datasets;